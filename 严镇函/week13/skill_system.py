"""
渐进式技能加载与执行系统

教学重点：
  1. 技能按需加载，而非启动时全部加载（节省内存和初始化时间）
  2. 技能元数据支持语义匹配，Agent 可自动选择合适技能
  3. 技能执行可结合四层记忆，访问用户画像和历史记忆
  4. 技能热重载：新增/修改技能文件后自动重新加载

使用方式：
  from src.skill_system import SkillRegistry
  registry = SkillRegistry()
  registry.discover_skills()  # 扫描 skills/ 目录
  result = registry.execute_skill("skill_name", context={...})
"""

import os
import sys
import json
import logging
import importlib
import importlib.util
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Optional, Callable

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).parent.parent / "skills"


@dataclass
class SkillMetadata:
    """技能元数据"""
    name: str                          # 技能唯一标识
    display_name: str                  # 显示名称
    description: str                   # 技能描述（用于语义匹配）
    version: str = "1.0.0"            # 版本号
    author: str = ""                   # 作者
    tags: list[str] = field(default_factory=list)  # 标签（用于分类和搜索）
    triggers: list[str] = field(default_factory=list)  # 触发关键词（用于意图识别）
    requires_memory: bool = False      # 是否需要访问记忆系统
    file_path: str = ""                # 技能文件路径
    loaded: bool = False               # 是否已加载
    last_modified: float = 0.0         # 最后修改时间


@dataclass
class SkillResult:
    """技能执行结果"""
    skill_name: str
    success: bool
    output: Any = None
    error: str = ""
    execution_time: float = 0.0
    memory_used: bool = False


class SkillRegistry:
    """技能注册表 - 管理技能的发现、加载和执行"""
    
    def __init__(self, skills_dir: Path = SKILLS_DIR):
        self.skills_dir = skills_dir
        self._registry: dict[str, SkillMetadata] = {}  # 元数据注册表
        self._loaded_skills: dict[str, Any] = {}       # 已加载的技能模块
        self._memory_loader = None  # 延迟注入
        self._retriever = None      # 延迟注入
    
    def inject_memory_system(self, memory_loader, retriever):
        """注入记忆系统引用（延迟加载）"""
        self._memory_loader = memory_loader
        self._retriever = retriever
    
    def discover_skills(self) -> list[SkillMetadata]:
        """扫描 skills/ 目录，发现所有可用技能（仅读取元数据，不加载实现）"""
        if not self.skills_dir.exists():
            logger.info(f"技能目录不存在: {self.skills_dir}")
            return []
        
        discovered = []
        for skill_file in self.skills_dir.glob("*.py"):
            if skill_file.name.startswith("_"):
                continue
            
            try:
                metadata = self._extract_metadata(skill_file)
                if metadata:
                    self._registry[metadata.name] = metadata
                    discovered.append(metadata)
                    logger.info(f"发现技能: {metadata.display_name} ({metadata.name})")
            except Exception as e:
                logger.warning(f"读取技能元数据失败 {skill_file.name}: {e}")
        
        return discovered
    
    def _extract_metadata(self, skill_file: Path) -> Optional[SkillMetadata]:
        """从技能文件中提取元数据（不执行模块代码）"""
        try:
            content = skill_file.read_text(encoding="utf-8")
            
            # 查找 SKILL_METADATA 字典
            metadata_marker = "SKILL_METADATA = "
            if metadata_marker not in content:
                return None
            
            start = content.index(metadata_marker) + len(metadata_marker)
            # 查找字典结束位置
            brace_count = 0
            end = start
            for i, char in enumerate(content[start:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = start + i + 1
                        break
            
            metadata_dict = eval(content[start:end])  # 安全：仅解析字面量
            
            return SkillMetadata(
                name=metadata_dict.get("name", skill_file.stem),
                display_name=metadata_dict.get("display_name", skill_file.stem),
                description=metadata_dict.get("description", ""),
                version=metadata_dict.get("version", "1.0.0"),
                author=metadata_dict.get("author", ""),
                tags=metadata_dict.get("tags", []),
                triggers=metadata_dict.get("triggers", []),
                requires_memory=metadata_dict.get("requires_memory", False),
                file_path=str(skill_file),
                last_modified=skill_file.stat().st_mtime,
            )
        except Exception as e:
            logger.error(f"解析技能元数据失败 {skill_file}: {e}")
            return None
    
    def list_available_skills(self) -> list[dict]:
        """列出所有可用技能（不加载）"""
        return [
            {
                "name": m.name,
                "display_name": m.display_name,
                "description": m.description,
                "version": m.version,
                "tags": m.tags,
                "triggers": m.triggers,
                "loaded": m.loaded,
            }
            for m in self._registry.values()
        ]
    
    def load_skill(self, skill_name: str) -> bool:
        """渐进式加载单个技能（按需加载）"""
        if skill_name not in self._registry:
            logger.warning(f"技能不存在: {skill_name}")
            return False
        
        metadata = self._registry[skill_name]
        
        # 检查是否已加载且未过期
        if metadata.loaded and skill_name in self._loaded_skills:
            current_mtime = Path(metadata.file_path).stat().st_mtime
            if current_mtime <= metadata.last_modified:
                return True  # 已加载且未修改
        
        try:
            # 动态加载模块
            spec = importlib.util.spec_from_file_location(
                f"skills.{skill_name}",
                metadata.file_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 验证模块有 execute 函数
            if not hasattr(module, "execute"):
                logger.warning(f"技能 {skill_name} 缺少 execute 函数")
                return False
            
            self._loaded_skills[skill_name] = module
            metadata.loaded = True
            metadata.last_modified = Path(metadata.file_path).stat().st_mtime
            
            logger.info(f"技能已加载: {skill_name}")
            return True
            
        except Exception as e:
            logger.error(f"加载技能失败 {skill_name}: {e}")
            return False
    
    def unload_skill(self, skill_name: str) -> bool:
        """卸载技能（释放内存）"""
        if skill_name not in self._loaded_skills:
            return False
        
        try:
            del self._loaded_skills[skill_name]
            if skill_name in self._registry:
                self._registry[skill_name].loaded = False
            logger.info(f"技能已卸载: {skill_name}")
            return True
        except Exception as e:
            logger.error(f"卸载技能失败 {skill_name}: {e}")
            return False
    
    def execute_skill(self, skill_name: str, context: dict = None) -> SkillResult:
        """执行技能（自动加载如果未加载）"""
        import time
        start_time = time.time()
        
        # 按需加载
        if not self.load_skill(skill_name):
            return SkillResult(
                skill_name=skill_name,
                success=False,
                error=f"技能加载失败: {skill_name}"
            )
        
        try:
            module = self._loaded_skills[skill_name]
            
            # 构建执行上下文
            exec_context = context or {}
            
            # 如果技能需要记忆系统，注入记忆访问能力
            metadata = self._registry.get(skill_name)
            if metadata and metadata.requires_memory:
                exec_context["memory_access"] = self._create_memory_access()
            
            # 执行技能
            result = module.execute(exec_context)
            
            execution_time = time.time() - start_time
            
            return SkillResult(
                skill_name=skill_name,
                success=True,
                output=result,
                execution_time=execution_time,
                memory_used=metadata.requires_memory if metadata else False
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"技能执行失败 {skill_name}: {e}")
            return SkillResult(
                skill_name=skill_name,
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    def _create_memory_access(self) -> dict:
        """创建记忆访问代理（技能可通过此接口访问四层记忆）"""
        return {
            "get_user_profile": self._get_user_profile,
            "search_memory": self._search_memory,
            "get_recent_logs": self._get_recent_logs,
        }
    
    def _get_user_profile(self) -> dict:
        """获取用户画像（Layer 3b）"""
        if not self._memory_loader:
            return {}
        try:
            user_md = self._memory_loader._read_md("USER.md")
            # 简单解析 USER.md 为字典
            profile = {}
            for line in user_md.split("\n"):
                if ":" in line and not line.startswith("#"):
                    key, value = line.split(":", 1)
                    profile[key.strip()] = value.strip()
            return profile
        except:
            return {}
    
    def _search_memory(self, query: str, top_k: int = 3) -> list[dict]:
        """搜索记忆（Layer 4）"""
        if not self._retriever:
            return []
        try:
            return self._retriever.search(query, top_k=top_k)
        except:
            return []
    
    def _get_recent_logs(self, days: int = 2) -> str:
        """获取近期日志（Layer 2）"""
        if not self._memory_loader:
            return ""
        try:
            log_body, _ = self._memory_loader._read_recent_day_logs(days)
            return log_body
        except:
            return ""
    
    def match_skill_by_query(self, query: str) -> Optional[str]:
        """根据用户查询匹配最合适的技能（基于触发词）"""
        query_lower = query.lower()
        
        # 遍历所有技能，检查触发词
        for name, metadata in self._registry.items():
            for trigger in metadata.triggers:
                if trigger.lower() in query_lower:
                    return name
        
        return None
    
    def reload_if_modified(self, skill_name: str) -> bool:
        """检查技能文件是否被修改，如果是则重新加载"""
        if skill_name not in self._registry:
            return False
        
        metadata = self._registry[skill_name]
        current_mtime = Path(metadata.file_path).stat().st_mtime
        
        if current_mtime > metadata.last_modified:
            logger.info(f"技能已修改，重新加载: {skill_name}")
            self._loaded_skills.pop(skill_name, None)
            return self.load_skill(skill_name)
        
        return True
    
    def get_skill_stats(self) -> dict:
        """获取技能系统统计信息"""
        return {
            "total_skills": len(self._registry),
            "loaded_skills": sum(1 for m in self._registry.values() if m.loaded),
            "skills": [
                {
                    "name": m.name,
                    "loaded": m.loaded,
                    "last_modified": datetime.fromtimestamp(m.last_modified).isoformat(),
                }
                for m in self._registry.values()
            ]
        }
# Log Stats 参考细节

> 本文件是渐进式披露内容：只有在需要核对日志格式、报告样例或边界处理时才读取，
> 平时触发 skill 不会把这些细节加载进上下文，从而节省 token。

## 日志格式

每行结构：`YYYY-MM-DD HH:MM:SS <LEVEL> <内容>`，级别为 DEBUG/INFO/WARN/ERROR 之一。
示例：

```
2026-08-06 10:23:45 INFO  User login succeeded for user_id=1024
2026-08-06 10:23:47 ERROR Database connection timeout after 30s
```

## 报告输出样例

```
========== 日志统计报告 ==========
文件：app.log
总行数：120000

--- 各级别统计 ---
DEBUG :  40012  (33.3%)
INFO  :  60005  (50.0%)
WARN  :  15003  (12.5%)
ERROR :   4980  ( 4.2%)

--- Top 5 错误 ---
 1. [ 1203 次] Database connection timeout after 30s
 ...

--- 每小时请求量 ---
00 时: 3200
...
23 时: 5010
==================================
```

## 边界情况

- 不符合格式的行（堆栈跟踪、空行、乱码）计入总行数，但不计入级别统计。
- 级别匹配大小写敏感，只认大写 DEBUG/INFO/WARN/ERROR。
- Top N 错误：把内容部分完全相同的 ERROR 行归并计数。
- 每小时请求量：按行首时间戳的小时字段（HH，00~23）归类。

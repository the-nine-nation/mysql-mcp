# MySQL Readonly MCP Server

这是一个基于MCP（Model Context Protocol）的MySQL只读查询服务器，它允许通过安全的只读查询接口访问MySQL数据库。

## 功能特点

- 支持MySQL数据库的只读查询操作
- 使用连接池管理数据库连接
- 自动限制查询结果数量
- 支持参数化查询
- 提供格式化的查询结果输出
- 内置安全检查，防止非只读操作

## 环境要求

- Python 3.7+
- MySQL 5.7+
- aiomysql

## 安装

1. 克隆仓库：
```bash
git clone [repository-url]
cd mysql_readonly_mcp
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```



## 使用方法

以cursor为例,将如下字典放入config.json: 其中 sys.executable为python虚拟环境的执行文件,通常名字为python,conda或uv下皆可以
mysql_mcp_py为main.py文件的绝对路径


```json
    "mysql": {
        "command": sys.executable,
        "args": [mysql_mcp_py],
        "env": {
            "MYSQL_ENABLED": "true",
            "MYSQL_HOST": "mysql数据库的ip",
            "MYSQL_PORT": "mysql数据库的端口",
            "MYSQL_DATABASE": "mysql数据库的名称",
            "MYSQL_USERNAME": "mysql数据库的用户名",
            "MYSQL_PASSWORD": "mysql数据库的密码",
            "MYSQL_POOL_MINSIZE": "1",
            "MYSQL_POOL_MAXSIZE": "10",
            "MYSQL_RESOURCE_DESC_FILE": "mysql数据库的资源描述文件路径"
        }
    }
```

请注意: 1.MYSQL_RESOURCE_DESC_FILE是一个说明,可以将数据库中一些信息放入其中,例如什么表是做什么用的,能够提升模型理解能力. 
2.MYSQL_ENABLED默认可以不用填

## 安全特性

- 只允许只读操作
- 防止多语句执行
- 自动限制查询结果数量
- 使用参数化查询防止SQL注入

## 注意事项

- 确保数据库用户只有只读权限
- 资源描述文件是必需的，用于定义可用的查询接口
- 查询结果默认限制为20行，可通过环境变量调整

## 贡献

欢迎提交Issue和Pull Request。

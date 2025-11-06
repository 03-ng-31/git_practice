"""
MCP Server Project Structure
============================

project_root/
├── config.py           # Centralized settings loaded from env vars
├── run.py              # Entry point that starts the ASGI server
├── main.py             # Builds FastAPI app and mounts MCP server
├── app/
│   ├── __init__.py     # Exposes create_mcp_server
│   └── mcp_server.py   # Defines MCP server and its tools
├── requirements.txt    # Python dependencies
├── .env.dev            # Sample environment for development
├── .env.test           # Sample environment for testing
├── .env.prod           # Sample environment for production
└── README.md           # Project documentation and usage notes

"""

# ==================== config.py ====================
"""Centralized settings loaded from environment variables"""

import os
from enum import Enum
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Environment(str, Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Environment
    ENVIRONMENT: Environment = Field(default=Environment.DEVELOPMENT)
    DEBUG: bool = Field(default=False)
    
    # Application metadata
    APP_NAME: str = Field(default="Custom MCP Server")
    APP_VERSION: str = Field(default="1.0.0")
    
    # Server configuration
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    BASE_PATH: str = Field(default="")
    
    # MCP server metadata
    MCP_SERVER_NAME: str = Field(default="custom-mcp-server")
    MCP_SERVER_VERSION: str = Field(default="1.0.0")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO")
    
    # CORS settings
    CORS_ORIGINS: List[str] = Field(default=["*"])
    
    # Optional external services
    DATABASE_URL: Optional[str] = Field(default=None)
    API_KEY: Optional[str] = Field(default=None)
    REDIS_URL: Optional[str] = Field(default=None)
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.ENVIRONMENT == Environment.DEVELOPMENT
    
    @property
    def is_test(self) -> bool:
        """Check if running in test mode"""
        return self.ENVIRONMENT == Environment.TEST
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.ENVIRONMENT == Environment.PRODUCTION


def load_settings() -> Settings:
    """
    Load settings based on ENVIRONMENT variable.
    Automatically loads the correct .env file.
    """
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    # Map environment to .env file
    env_files = {
        "development": ".env.dev",
        "dev": ".env.dev",
        "test": ".env.test",
        "testing": ".env.test",
        "production": ".env.prod",
        "prod": ".env.prod",
    }
    
    env_file = env_files.get(env, ".env.dev")
    
    # Load settings from the appropriate env file
    if os.path.exists(env_file):
        return Settings(_env_file=env_file)
    else:
        # Fallback to environment variables only
        return Settings()


# Global settings instance
settings = load_settings()


# ==================== app/mcp_server.py ====================
"""Defines MCP server and its tools"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from fastmcp import FastMCP
import logging

logger = logging.getLogger(__name__)


# In-memory task storage (replace with database in production)
TASKS_STORE: List[Dict[str, Any]] = [
    {
        "id": "1",
        "title": "Setup MCP Server",
        "description": "Deploy custom MCP server to cloud infrastructure",
        "status": "completed",
        "priority": "high",
        "created_at": "2024-01-15T10:00:00Z",
        "updated_at": "2024-01-15T15:30:00Z"
    },
    {
        "id": "2",
        "title": "Integrate FastAPI",
        "description": "Add FastAPI integration with fastmcp",
        "status": "in_progress",
        "priority": "high",
        "created_at": "2024-01-16T09:30:00Z",
        "updated_at": "2024-01-16T14:20:00Z"
    },
    {
        "id": "3",
        "title": "Configure base path",
        "description": "Handle automatic base URL prefix in cloud deployment",
        "status": "pending",
        "priority": "medium",
        "created_at": "2024-01-17T14:20:00Z",
        "updated_at": "2024-01-17T14:20:00Z"
    }
]


def create_mcp_server(name: str, version: str) -> FastMCP:
    """
    Create and configure the MCP server with all tools.
    
    Args:
        name: MCP server name
        version: MCP server version
        
    Returns:
        Configured FastMCP instance
    """
    logger.info(f"Creating MCP server: {name} v{version}")
    
    # Initialize FastMCP server
    mcp = FastMCP(name=name, version=version)
    
    # Register all tools
    _register_tools(mcp)
    
    logger.info(f"MCP server created with {len(_get_tool_names())} tools")
    return mcp


def _register_tools(mcp: FastMCP) -> None:
    """Register all MCP tools"""
    
    @mcp.tool()
    def fetch_tasks(
        status: Optional[str] = None,
        priority: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fetch tasks from the task management system.
        
        Args:
            status: Filter by status (pending, in_progress, completed)
            priority: Filter by priority (low, medium, high)
            limit: Maximum number of tasks to return (default: 10)
            
        Returns:
            List of tasks matching the criteria
        """
        try:
            filtered = TASKS_STORE.copy()
            
            if status:
                filtered = [t for t in filtered if t["status"] == status]
            
            if priority:
                filtered = [t for t in filtered if t["priority"] == priority]
            
            result = filtered[:limit]
            logger.debug(f"Fetched {len(result)} tasks (status={status}, priority={priority})")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching tasks: {e}")
            return {"error": str(e)}
    
    @mcp.tool()
    def get_task_by_id(task_id: str) -> Dict[str, Any]:
        """
        Get a specific task by ID.
        
        Args:
            task_id: The unique identifier of the task
            
        Returns:
            Task details or error message
        """
        try:
            task = next((t for t in TASKS_STORE if t["id"] == task_id), None)
            
            if task:
                logger.debug(f"Retrieved task: {task_id}")
                return task
            else:
                logger.warning(f"Task not found: {task_id}")
                return {"error": f"Task with ID {task_id} not found"}
                
        except Exception as e:
            logger.error(f"Error getting task {task_id}: {e}")
            return {"error": str(e)}
    
    @mcp.tool()
    def create_task(
        title: str,
        description: str,
        priority: str = "medium"
    ) -> Dict[str, Any]:
        """
        Create a new task.
        
        Args:
            title: Task title (required)
            description: Task description (required)
            priority: Task priority - low, medium, or high (default: medium)
            
        Returns:
            Created task details
        """
        try:
            if priority not in ["low", "medium", "high"]:
                return {"error": "Priority must be one of: low, medium, high"}
            
            new_id = str(len(TASKS_STORE) + 1)
            now = datetime.utcnow().isoformat() + "Z"
            
            new_task = {
                "id": new_id,
                "title": title,
                "description": description,
                "status": "pending",
                "priority": priority,
                "created_at": now,
                "updated_at": now
            }
            
            TASKS_STORE.append(new_task)
            logger.info(f"Task created: {new_id} - {title}")
            return new_task
            
        except Exception as e:
            logger.error(f"Error creating task: {e}")
            return {"error": str(e)}
    
    @mcp.tool()
    def update_task_status(task_id: str, status: str) -> Dict[str, Any]:
        """
        Update the status of a task.
        
        Args:
            task_id: The unique identifier of the task
            status: New status - pending, in_progress, or completed
            
        Returns:
            Updated task details or error message
        """
        try:
            if status not in ["pending", "in_progress", "completed"]:
                return {"error": "Status must be one of: pending, in_progress, completed"}
            
            task = next((t for t in TASKS_STORE if t["id"] == task_id), None)
            
            if task:
                task["status"] = status
                task["updated_at"] = datetime.utcnow().isoformat() + "Z"
                logger.info(f"Task {task_id} status updated to: {status}")
                return task
            else:
                return {"error": f"Task with ID {task_id} not found"}
                
        except Exception as e:
            logger.error(f"Error updating task {task_id}: {e}")
            return {"error": str(e)}
    
    @mcp.tool()
    def delete_task(task_id: str) -> Dict[str, Any]:
        """
        Delete a task.
        
        Args:
            task_id: The unique identifier of the task to delete
            
        Returns:
            Success message or error
        """
        try:
            task = next((t for t in TASKS_STORE if t["id"] == task_id), None)
            
            if task:
                TASKS_STORE.remove(task)
                logger.info(f"Task deleted: {task_id}")
                return {
                    "success": True,
                    "message": f"Task {task_id} deleted successfully"
                }
            else:
                return {
                    "success": False,
                    "error": f"Task with ID {task_id} not found"
                }
                
        except Exception as e:
            logger.error(f"Error deleting task {task_id}: {e}")
            return {"error": str(e)}
    
    @mcp.tool()
    def get_task_statistics() -> Dict[str, Any]:
        """
        Get statistics about all tasks.
        
        Returns:
            Task statistics including total count and breakdown by status/priority
        """
        try:
            total = len(TASKS_STORE)
            by_status = {}
            by_priority = {}
            
            for task in TASKS_STORE:
                status = task["status"]
                priority = task["priority"]
                
                by_status[status] = by_status.get(status, 0) + 1
                by_priority[priority] = by_priority.get(priority, 0) + 1
            
            stats = {
                "total_tasks": total,
                "by_status": by_status,
                "by_priority": by_priority
            }
            
            logger.debug("Task statistics retrieved")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting task statistics: {e}")
            return {"error": str(e)}


def _get_tool_names() -> List[str]:
    """Get list of registered tool names"""
    return [
        "fetch_tasks",
        "get_task_by_id",
        "create_task",
        "update_task_status",
        "delete_task",
        "get_task_statistics"
    ]


# ==================== app/__init__.py ====================
"""Exposes create_mcp_server"""

from .mcp_server import create_mcp_server

__all__ = ["create_mcp_server"]


# ==================== main.py ====================
"""Builds FastAPI app and mounts MCP server"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from app import create_mcp_server

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Build and configure the FastAPI application.
    
    Returns:
        Configured FastAPI instance
    """
    logger.info(f"Building FastAPI app for {settings.ENVIRONMENT.value} environment")
    
    # Determine URL paths based on BASE_PATH
    docs_url = f"{settings.BASE_PATH}/docs" if settings.BASE_PATH else "/docs"
    redoc_url = f"{settings.BASE_PATH}/redoc" if settings.BASE_PATH else "/redoc"
    openapi_url = f"{settings.BASE_PATH}/openapi.json" if settings.BASE_PATH else "/openapi.json"
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    logger.info(f"CORS configured: {settings.CORS_ORIGINS}")
    
    # Create MCP server
    mcp = create_mcp_server(
        name=settings.MCP_SERVER_NAME,
        version=settings.MCP_SERVER_VERSION
    )
    
    # Mount MCP server to FastAPI
    mount_path = settings.BASE_PATH if settings.BASE_PATH else ""
    logger.info(f"Mounting MCP server at path: '{mount_path}'")
    mcp.integrate_fastapi(app, path=mount_path)
    
    # Add health and info endpoints
    _add_endpoints(app)
    
    logger.info("FastAPI app built successfully")
    return app


def _add_endpoints(app: FastAPI) -> None:
    """Add health check and info endpoints"""
    
    health_path = f"{settings.BASE_PATH}/health" if settings.BASE_PATH else "/health"
    info_path = f"{settings.BASE_PATH}/info" if settings.BASE_PATH else "/info"
    
    @app.get(health_path)
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT.value
        }
    
    @app.get(info_path)
    async def server_info():
        """Server information endpoint"""
        base = settings.BASE_PATH or "/"
        return {
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "mcp_version": settings.MCP_SERVER_VERSION,
            "environment": settings.ENVIRONMENT.value,
            "debug_mode": settings.DEBUG,
            "base_path": base,
            "endpoints": {
                "mcp": base,
                "docs": f"{settings.BASE_PATH}/docs" if settings.BASE_PATH else "/docs",
                "health": health_path,
                "info": info_path
            }
        }
    
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": f"Welcome to {settings.APP_NAME}",
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT.value,
            "mcp_endpoint": settings.BASE_PATH or "/",
            "documentation": f"{settings.BASE_PATH}/docs" if settings.BASE_PATH else "/docs",
            "health_check": health_path
        }
    
    logger.info("Health and info endpoints registered")


# ==================== run.py ====================
"""Entry point that starts the ASGI server"""

import sys
import uvicorn
from config import settings
from main import create_app


def print_startup_banner() -> None:
    """Print startup banner with configuration details"""
    
    # Determine environment display
    env_icons = {
        "development": "🔧",
        "test": "🧪",
        "production": "🚀"
    }
    env_icon = env_icons.get(settings.ENVIRONMENT.value, "⚙️")
    
    # Determine URLs
    if settings.is_development:
        base_url = f"http://{settings.HOST}:{settings.PORT}"
    elif settings.is_test:
        base_url = f"http://{settings.HOST}:{settings.PORT}"
    else:
        base_url = "https://gb-gateway.att.com/domain"
    
    base_path = settings.BASE_PATH or "/"
    
    print(f"""
╔════════════════════════════════════════════════════════════════╗
║  {settings.APP_NAME} v{settings.APP_VERSION}
║  
║  Environment: {env_icon} {settings.ENVIRONMENT.value.upper()}
║  Debug Mode: {'✓ Enabled' if settings.DEBUG else '✗ Disabled'}
║  Log Level: {settings.LOG_LEVEL}
║  
║  🌐 Server Configuration:
║  ├─ Host: {settings.HOST}
║  ├─ Port: {settings.PORT}
║  └─ Base Path: {base_path}
║  
║  📡 Endpoints:
║  ├─ MCP Server: {base_url}{base_path}
║  ├─ API Docs:   {base_url}{settings.BASE_PATH}/docs" if settings.BASE_PATH else "/docs"}
║  ├─ Health:     {base_url}{settings.BASE_PATH}/health" if settings.BASE_PATH else "/health"}
║  └─ Info:       {base_url}{settings.BASE_PATH}/info" if settings.BASE_PATH else "/info"}
║  
║  Press Ctrl+C to stop the server
╚════════════════════════════════════════════════════════════════╝
    """)


def main() -> None:
    """Main entry point"""
    try:
        # Create FastAPI app
        app = create_app()
        
        # Print startup banner
        print_startup_banner()
        
        # Start ASGI server
        uvicorn.run(
            app,
            host=settings.HOST,
            port=settings.PORT,
            log_level=settings.LOG_LEVEL.lower(),
            access_log=settings.DEBUG
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


# ==================== requirements.txt ====================
fastapi==0.104.1
fastmcp==0.1.0
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
python-dotenv==1.0.0


# ==================== .env.dev ====================
# Development Environment Configuration
ENVIRONMENT=development

# Application
APP_NAME=Custom MCP Server
APP_VERSION=1.0.0

# Server Settings
HOST=localhost
PORT=3020
BASE_PATH=

# Debug
DEBUG=true
LOG_LEVEL=DEBUG

# MCP Configuration
MCP_SERVER_NAME=custom-mcp-server-dev
MCP_SERVER_VERSION=1.0.0

# CORS (allow all for development)
CORS_ORIGINS=["*"]

# Optional: Database
# DATABASE_URL=sqlite:///./dev_database.db

# Optional: API Keys
# API_KEY=dev-api-key-here


# ==================== .env.test ====================
# Test Environment Configuration
ENVIRONMENT=test

# Application
APP_NAME=Custom MCP Server
APP_VERSION=1.0.0

# Server Settings
HOST=localhost
PORT=8001
BASE_PATH=/test/vagas/apps/mcp

# Debug
DEBUG=true
LOG_LEVEL=DEBUG

# MCP Configuration
MCP_SERVER_NAME=custom-mcp-server-test
MCP_SERVER_VERSION=1.0.0

# CORS
CORS_ORIGINS=["http://localhost:*","http://testserver"]

# Optional: Database
# DATABASE_URL=sqlite:///./test_database.db

# Optional: API Keys
# API_KEY=test-api-key-here


# ==================== .env.prod ====================
# Production Environment Configuration
ENVIRONMENT=production

# Application
APP_NAME=Custom MCP Server
APP_VERSION=1.0.0

# Server Settings
HOST=0.0.0.0
PORT=8000
BASE_PATH=/vagas/apps/mcp

# Debug
DEBUG=false
LOG_LEVEL=INFO

# MCP Configuration
MCP_SERVER_NAME=custom-mcp-server
MCP_SERVER_VERSION=1.0.0

# CORS (restrict to specific domains)
CORS_ORIGINS=["https://gb-gateway.att.com","https://*.att.com"]

# Production: Set these via environment variables, not in .env file
# DATABASE_URL=postgresql://user:password@host:5432/dbname
# API_KEY=your-production-api-key-here
# REDIS_URL=redis://redis-host:6379/0


# ==================== README.md ====================
# Custom MCP Server

A Model Context Protocol (MCP) server built with FastMCP and FastAPI, designed for cloud infrastructure deployment with multi-environment support.

## Project Structure

```
project_root/
├── config.py           # Centralized settings loaded from env vars
├── run.py              # Entry point that starts the ASGI server
├── main.py             # Builds FastAPI app and mounts MCP server
├── app/
│   ├── __init__.py     # Exposes create_mcp_server
│   └── mcp_server.py   # Defines MCP server and its tools
├── requirements.txt    # Python dependencies
├── .env.dev            # Sample environment for development
├── .env.test           # Sample environment for testing
├── .env.prod           # Sample environment for production
└── README.md           # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Choose Environment

Set the `ENVIRONMENT` variable to load the appropriate configuration:

```bash
# Development (localhost:3020, no base path)
export ENVIRONMENT=development

# Test (localhost:8001, test base path)
export ENVIRONMENT=test

# Production (0.0.0.0:8000, full base path)
export ENVIRONMENT=production
```

### 3. Run Server

```bash
python run.py
```

## Environment Configurations

### Development Environment
- **Host**: `localhost`
- **Port**: `3020`
- **Base Path**: None (root level)
- **URL**: `http://localhost:3020`
- **Debug**: Enabled
- **Log Level**: DEBUG
- **CORS**: Allow all origins

**Access Points:**
- MCP Server: http://localhost:3020
- API Docs: http://localhost:3020/docs
- Health: http://localhost:3020/health

### Test Environment
- **Host**: `localhost`
- **Port**: `8001`
- **Base Path**: `/test/vagas/apps/mcp`
- **URL**: `http://localhost:8001/test/vagas/apps/mcp`
- **Debug**: Enabled
- **Log Level**: DEBUG
- **CORS**: Restricted to localhost

**Access Points:**
- MCP Server: http://localhost:8001/test/vagas/apps/mcp
- API Docs: http://localhost:8001/test/vagas/apps/mcp/docs
- Health: http://localhost:8001/test/vagas/apps/mcp/health

### Production Environment
- **Host**: `0.0.0.0`
- **Port**: `8000`
- **Base Path**: `/vagas/apps/mcp`
- **Cloud URL**: `https://gb-gateway.att.com/domain/vagas/apps/mcp`
- **Debug**: Disabled
- **Log Level**: INFO
- **CORS**: Restricted to specific domains

**Access Points:**
- MCP Server: https://gb-gateway.att.com/domain/vagas/apps/mcp
- API Docs: https://gb-gateway.att.com/domain/vagas/apps/mcp/docs
- Health: https://gb-gateway.att.com/domain/vagas/apps/mcp/health

## Available MCP Tools

The server provides 6 task management tools:

### 1. fetch_tasks
Retrieve tasks with optional filtering.

**Parameters:**
- `status` (optional): Filter by status (pending, in_progress, completed)
- `priority` (optional): Filter by priority (low, medium, high)
- `limit` (optional): Maximum tasks to return (default: 10)

**Example:**
```python
fetch_tasks(status="in_progress", priority="high", limit=5)
```

### 2. get_task_by_id
Get a specific task by its ID.

**Parameters:**
- `task_id` (required): The unique task identifier

**Example:**
```python
get_task_by_id(task_id="1")
```

### 3. create_task
Create a new task.

**Parameters:**
- `title` (required): Task title
- `description` (required): Task description
- `priority` (optional): Priority level (low, medium, high, default: medium)

**Example:**
```python
create_task(
    title="Implement authentication",
    description="Add JWT-based authentication to API",
    priority="high"
)
```

### 4. update_task_status
Update a task's status.

**Parameters:**
- `task_id` (required): Task identifier
- `status` (required): New status (pending, in_progress, completed)

**Example:**
```python
update_task_status(task_id="2", status="completed")
```

### 5. delete_task
Delete a task.

**Parameters:**
- `task_id` (required): Task identifier to delete

**Example:**
```python
delete_task(task_id="3")
```

### 6. get_task_statistics
Get task statistics and summaries.

**No parameters required.**

**Example:**
```python
get_task_statistics()
```

**Returns:**
```json
{
  "total_tasks": 3,
  "by_status": {
    "pending": 1,
    "in_progress": 1,
    "completed": 1
  },
  "by_priority": {
    "high": 2,
    "medium": 1
  }
}
```

## Configuration

### Environment Variables

All configuration is managed through environment variables, loaded from `.env.*` files:

| Variable | Description | Dev | Test | Prod |
|----------|-------------|-----|------|------|
| `ENVIRONMENT` | Environment name | development | test | production |
| `HOST` | Server host | localhost | localhost | 0.0.0.0 |
| `PORT` | Server port | 3020 | 8001 | 8000 |
| `BASE_PATH` | URL base path | (empty) | /test/vagas/apps/mcp | /vagas/apps/mcp |
| `DEBUG` | Debug mode | true | true | false |
| `LOG_LEVEL` | Logging level | DEBUG | DEBUG | INFO |
| `CORS_ORIGINS` | Allowed origins | ["*"] | ["http://localhost:*"] | ["https://gb-gateway.att.com"] |

### Customizing Configuration

1. **For development**: Edit `.env.dev`
2. **For testing**: Edit `.env.test`
3. **For production**: Use environment variables (don't store secrets in `.env.prod`)

## Development Workflow

### Local Development

```bash
# Set environment
export ENVIRONMENT=development

# Run server with auto-reload
python run.py

# Or use uvicorn directly with reload
uvicorn main:create_app --reload --host localhost --port 3020
```

### Testing MCP Tools

Access the interactive API docs to test tools:

```
http://localhost:3020/docs
```

### Adding New Tools

1. Open `app/mcp_server.py`
2. Add your tool function inside `_register_tools()`:

```python
@mcp.tool()
def my_new_tool(param: str) -> dict:
    """
    Description of what the tool does.
    
    Args:
        param: Parameter description
        
    Returns:
        Return value description
    """
    try:
        # Your logic here
        result = {"data": param}
        logger.info(f"Tool executed: {param}")
        return result
    except Exception as e:
        logger.error(f"Tool error: {e}")
        return {"error": str(e)}
```

3. Update `_get_tool_names()` list
4. Restart the server

## Deployment

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Set default environment (override at runtime)
ENV ENVIRONMENT=production

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "run.py"]
```

Build and run:

```bash
# Build image
docker build -t mcp-server:latest .

# Run development
docker run -p 3020:3020 -e ENVIRONMENT=development mcp-server:latest

# Run production
docker run -p 8000:8000 -e ENVIRONMENT=production \
  -e DATABASE_URL=postgres://... \
  -e API_KEY=your-key \
  mcp-server:latest
```

### Cloud Deployment

#### Prerequisites
- Set `ENVIRONMENT=production`
- Configure environment variables in your cloud platform
- Ensure health check endpoint is configured: `/vagas/apps/mcp/health`

#### AWS ECS/Fargate

Task definition environment variables:
```json
{
  "environment": [
    {"name": "ENVIRONMENT", "value": "production"},
    {"name": "HOST", "value": "0.0.0.0"},
    {"name": "PORT", "value": "8000"},
    {"name": "BASE_PATH", "value": "/vagas/apps/mcp"},
    {"name": "
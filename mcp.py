"""
Dynamic Task Orchestrator with LangGraph - Simple Enhancement of Existing Agents
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# FastMCP and LangChain imports
from mcp.server.fastmcp import FastMCP
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ================================
# SIMPLE STATE FOR LANGGRAPH
# ================================

class TaskOrchestratorState(TypedDict):
    """Simple state that tracks our existing workflow"""
    # Input
    user_query: str
    user_id: str
    
    # Workflow decisions
    needs_structured_output: bool
    
    # Agent results
    task_data: Optional[str]
    inference_result: Optional[str]
    structured_result: Optional[str]
    
    # Final output
    final_response: str
    response_type: str  # "natural_language" or "structured_json"
    
    # Execution tracking
    execution_path: List[str]
    errors: List[str]

# ================================
# CONTEXT MANAGEMENT (SIMPLIFIED)
# ================================

class SimpleContext:
    """Simplified context management"""
    
    def __init__(self):
        self.system_stats = {
            "queries_processed": 0,
            "structured_outputs_generated": 0,
            "start_time": datetime.now().isoformat()
        }
        
        self.user_sessions = {}
    
    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get or create simple user context"""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                "cached_tasks": [],
                "query_count": 0,
                "last_activity": datetime.now().isoformat()
            }
        return self.user_sessions[user_id]
    
    def update_stats(self, stat_type: str):
        """Update system statistics"""
        if stat_type in self.system_stats:
            self.system_stats[stat_type] += 1

# Global context instance
context_manager = SimpleContext()

# ================================
# EXISTING FASTMCP SERVER (UNCHANGED)
# ================================

class TaskDataSource:
    """Task data source - same as before"""
    
    def __init__(self):
        self.mock_tasks = [
            {
                "id": "task_001",
                "title": "Complete Q4 Financial Report",
                "description": "Prepare comprehensive financial analysis for Q4",
                "status": "in_progress",
                "priority": "high",
                "assigned_to": "user_123",
                "created_date": "2024-01-01",
                "due_date": "2024-01-15",
                "completion_percentage": 60,
                "category": "finance"
            },
            {
                "id": "task_002", 
                "title": "Review Marketing Campaign Metrics",
                "description": "Analyze effectiveness of recent marketing campaigns",
                "status": "pending",
                "priority": "medium",
                "assigned_to": "user_123",
                "created_date": "2024-01-05",
                "due_date": "2024-01-20",
                "completion_percentage": 0,
                "category": "marketing"
            },
            {
                "id": "task_003",
                "title": "Update Customer Database", 
                "description": "Clean and update customer contact information",
                "status": "completed",
                "priority": "low",
                "assigned_to": "user_123",
                "created_date": "2023-12-28",
                "due_date": "2024-01-10", 
                "completion_percentage": 100,
                "category": "data"
            },
            {
                "id": "task_004",
                "title": "Prepare Team Meeting Agenda",
                "description": "Create agenda for weekly team sync meeting",
                "status": "overdue",
                "priority": "high",
                "assigned_to": "user_123",
                "created_date": "2024-01-08",
                "due_date": "2024-01-12",
                "completion_percentage": 0,
                "category": "management"
            }
        ]
    
    async def fetch_user_tasks_from_api(self, user_id: str) -> List[Dict]:
        """Integrate with your existing FetchTasksTool"""
        try:
            # Use your existing tool
            # from task_tools import FetchTasksTool
            # fetch_tool = FetchTasksTool()
            # result_json = fetch_tool._run(user_id=user_id)
            # return json.loads(result_json)
            
            # Mock for development
            await asyncio.sleep(0.1)
            return [task for task in self.mock_tasks if task["assigned_to"] == user_id]
            
        except Exception as e:
            logging.error(f"API call failed: {e}")
            return []

# Initialize data source
task_data_source = TaskDataSource()

# Create FastMCP server (same as before)
task_mcp = FastMCP("TaskManagementServer")

@task_mcp.tool()
async def task_agent(user_id: str, prompt: str) -> str:
    """
    Task Agent: Fetches task data from API (unchanged from your original design)
    """
    try:
        context_manager.update_stats("queries_processed")
        
        logging.info(f"Task Agent called for user: {user_id}")
        
        # Fetch tasks from your API
        tasks = await task_data_source.fetch_user_tasks_from_api(user_id)
        
        # Return task data in your expected format
        response = {
            "success": True,
            "user_id": user_id,
            "prompt": prompt,
            "task_data": tasks,
            "total_tasks": len(tasks),
            "fetch_timestamp": datetime.now().isoformat()
        }
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        logging.error(f"Task Agent error: {e}")
        error_response = {
            "success": False,
            "user_id": user_id,
            "prompt": prompt,
            "error": str(e),
            "fetch_timestamp": datetime.now().isoformat()
        }
        return json.dumps(error_response, indent=2)

# ================================
# OUR EXISTING AGENTS (SIMPLIFIED)
# ================================

class InferenceAgent:
    """
    Inference Agent: Processes user query with task data (unchanged from our design)
    """
    
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            api_key=openai_api_key
        )
        self.logger = logging.getLogger(__name__)
    
    async def process_query(self, user_query: str, task_data_json: str) -> str:
        """Process user query with task data"""
        try:
            # Parse task data
            task_data = json.loads(task_data_json)
            tasks = task_data.get("task_data", [])
            
            system_prompt = """You are a helpful task management assistant. Based on the provided task data, answer the user's query in a natural, conversational way.

Provide specific details about tasks when relevant including counts, statuses, priorities, due dates. Give actionable insights and recommendations. Be conversational and helpful."""

            user_message = f"""
User Query: {user_query}

Task Data:
{json.dumps(tasks, indent=2)}

Please provide a comprehensive answer to the user's query based on this task data.
"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]
            
            response = await self.llm.ainvoke(messages)
            return response.content
            
        except Exception as e:
            self.logger.error(f"Inference Agent error: {e}")
            return f"I encountered an error while processing your query: {str(e)}"

class StructuredOutputAgent:
    """
    Structured Output Agent: Generates JSON output (unchanged from our design)
    """
    
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            api_key=openai_api_key
        )
        self.logger = logging.getLogger(__name__)
    
    async def generate_structured_output(self, user_query: str, inference_output: str, task_data_json: str) -> str:
        """Generate structured JSON output"""
        try:
            context_manager.update_stats("structured_outputs_generated")
            
            # Parse task data
            task_data = json.loads(task_data_json)
            tasks = task_data.get("task_data", [])
            
            system_prompt = """You are a structured data generator that creates JSON responses for task management queries.

Generate well-structured JSON that includes:
- Summary statistics
- Relevant task details organized logically
- Insights and patterns from the inference output
- Actionable recommendations
- Proper categorization and formatting

Always return valid JSON only."""

            user_message = f"""
User Query: {user_query}
Inference Output: {inference_output}

Task Data:
{json.dumps(tasks, indent=2)}

Generate structured JSON output that addresses the user's request and incorporates insights from the inference output.
"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]
            
            response = await self.llm.ainvoke(messages)
            return response.content
            
        except Exception as e:
            self.logger.error(f"Structured Output Agent error: {e}")
            return json.dumps({
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

class StructuredOutputChecker:
    """Simple checker to see if structured output is needed"""
    
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            api_key=openai_api_key
        )
    
    async def needs_structured_output(self, user_query: str) -> bool:
        """Check if query needs structured output"""
        
        system_prompt = """Analyze if a user query requires structured JSON data output.

Queries that need structured output:
- "Generate report/dashboard/export"
- "Give me JSON data" 
- "Create structured overview"
- "Export my tasks"
- "Show data in JSON format"
- Requests for charts, graphs, visualizations
- Data analysis requests

Respond with only: true or false"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Does this query need structured JSON output? Query: {user_query}")
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            return "true" in response.content.lower()
        except:
            return False

# ================================
# LANGGRAPH ORCHESTRATOR (SIMPLE)
# ================================

class DynamicTaskOrchestrator:
    """
    LangGraph orchestrator that enhances our existing workflow with dynamic routing
    """
    
    def __init__(self, openai_api_key: str, mcp_server_path: str = None):
        self.openai_api_key = openai_api_key
        self.mcp_server_path = mcp_server_path or __file__
        
        # Initialize our existing agents
        self.inference_agent = InferenceAgent(openai_api_key)
        self.structured_output_agent = StructuredOutputAgent(openai_api_key)
        self.structured_checker = StructuredOutputChecker(openai_api_key)
        
        # LangGraph workflow
        self.workflow = None
        self.memory = MemorySaver()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def create_mcp_client(self) -> MultiServerMCPClient:
        """Create MCP client connection"""
        return MultiServerMCPClient({
            "task_server": {
                "command": "python",
                "args": [self.mcp_server_path, "mcp-server"],
                "transport": "stdio"
            }
        })
    
    def build_workflow(self):
        """Build simple LangGraph workflow"""
        
        # Create the workflow
        workflow = StateGraph(TaskOrchestratorState)
        
        # Add our workflow steps (matching your original design)
        workflow.add_node("check_structured_need", self._check_structured_need)
        workflow.add_node("fetch_task_data", self._fetch_task_data)  
        workflow.add_node("run_inference_agent", self._run_inference_agent)
        workflow.add_node("run_structured_agent", self._run_structured_agent)
        workflow.add_node("finalize_response", self._finalize_response)
        
        # Set entry point
        workflow.set_entry_point("check_structured_need")
        
        # Define the flow (your original workflow)
        workflow.add_edge("check_structured_need", "fetch_task_data")
        workflow.add_edge("fetch_task_data", "run_inference_agent")
        
        # Dynamic routing after inference
        workflow.add_conditional_edges(
            "run_inference_agent",
            self._should_generate_structured,
            {
                "structured": "run_structured_agent",
                "final": "finalize_response"
            }
        )
        
        workflow.add_edge("run_structured_agent", "finalize_response")
        workflow.add_edge("finalize_response", END)
        
        # Compile with memory
        self.workflow = workflow.compile(checkpointer=self.memory)
        
        self.logger.info("Simple dynamic workflow built successfully")
    
    async def _check_structured_need(self, state: TaskOrchestratorState) -> TaskOrchestratorState:
        """Step 1: Check if structured output is needed (parallel with your design)"""
        
        self.logger.info("Checking if structured output is needed...")
        
        try:
            needs_structured = await self.structured_checker.needs_structured_output(state["user_query"])
            state["needs_structured_output"] = needs_structured
            state["execution_path"].append("check_structured_need")
            
            self.logger.info(f"Structured output needed: {needs_structured}")
            
        except Exception as e:
            self.logger.error(f"Structured check failed: {e}")
            state["needs_structured_output"] = False
            state["errors"].append(f"Structured check failed: {str(e)}")
        
        return state
    
    async def _fetch_task_data(self, state: TaskOrchestratorState) -> TaskOrchestratorState:
        """Step 2: Fetch task data using task_agent (your original design)"""
        
        self.logger.info("Fetching task data...")
        
        try:
            async with self.create_mcp_client() as client:
                mcp_tools = await load_mcp_tools(client)
                
                # Use task_agent tool
                task_agent_tool = next(tool for tool in mcp_tools if tool.name == "task_agent")
                
                result = await task_agent_tool.ainvoke({
                    "user_id": state["user_id"],
                    "prompt": state["user_query"]
                })
                
                state["task_data"] = result
                state["execution_path"].append("fetch_task_data")
                
                self.logger.info("Task data fetched successfully")
                
        except Exception as e:
            self.logger.error(f"Task data fetch failed: {e}")
            state["errors"].append(f"Task fetch failed: {str(e)}")
            state["task_data"] = json.dumps({
                "success": False,
                "task_data": [],
                "error": str(e)
            })
        
        return state
    
    async def _run_inference_agent(self, state: TaskOrchestratorState) -> TaskOrchestratorState:
        """Step 3: Run inference agent (your original design)"""
        
        self.logger.info("Running Inference Agent...")
        
        try:
            result = await self.inference_agent.process_query(
                state["user_query"],
                state["task_data"]
            )
            
            state["inference_result"] = result
            state["execution_path"].append("run_inference_agent")
            
            self.logger.info("Inference Agent completed successfully")
            
        except Exception as e:
            self.logger.error(f"Inference Agent failed: {e}")
            state["errors"].append(f"Inference failed: {str(e)}")
            state["inference_result"] = "I encountered an error processing your request."
        
        return state
    
    def _should_generate_structured(self, state: TaskOrchestratorState) -> str:
        """Decision point: should we generate structured output?"""
        if state["needs_structured_output"]:
            return "structured"
        else:
            return "final"
    
    async def _run_structured_agent(self, state: TaskOrchestratorState) -> TaskOrchestratorState:
        """Step 4: Run structured output agent (your original design)"""
        
        self.logger.info("Running Structured Output Agent...")
        
        try:
            result = await self.structured_output_agent.generate_structured_output(
                state["user_query"],
                state["inference_result"],
                state["task_data"]
            )
            
            state["structured_result"] = result
            state["execution_path"].append("run_structured_agent")
            
            self.logger.info("Structured Output Agent completed successfully")
            
        except Exception as e:
            self.logger.error(f"Structured Output Agent failed: {e}")
            state["errors"].append(f"Structured output failed: {str(e)}")
            state["structured_result"] = json.dumps({
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
        
        return state
    
    async def _finalize_response(self, state: TaskOrchestratorState) -> TaskOrchestratorState:
        """Step 5: Finalize response (choose structured or natural language)"""
        
        self.logger.info("Finalizing response...")
        
        if state["needs_structured_output"] and state.get("structured_result"):
            state["final_response"] = state["structured_result"]
            state["response_type"] = "structured_json"
        else:
            state["final_response"] = state["inference_result"]
            state["response_type"] = "natural_language"
        
        state["execution_path"].append("finalize_response")
        
        self.logger.info(f"Response finalized: {state['response_type']}")
        
        return state
    
    async def process_query(self, user_query: str, user_id: str = "user_123") -> Dict[str, Any]:
        """Process query using LangGraph workflow"""
        
        try:
            # Build workflow if not built
            if not self.workflow:
                self.build_workflow()
            
            # Initialize state
            initial_state = {
                "user_query": user_query,
                "user_id": user_id,
                "needs_structured_output": False,
                "task_data": None,
                "inference_result": None,
                "structured_result": None,
                "final_response": "",
                "response_type": "natural_language",
                "execution_path": [],
                "errors": []
            }
            
            # Execute workflow
            config = {"configurable": {"thread_id": f"{user_id}_{datetime.now().timestamp()}"}}
            final_state = await self.workflow.ainvoke(initial_state, config)
            
            # Update user context
            user_context = context_manager.get_user_context(user_id)
            user_context["query_count"] += 1
            user_context["last_activity"] = datetime.now().isoformat()
            
            return {
                "success": True,
                "user_query": user_query,
                "user_id": user_id,
                "workflow_results": {
                    "needs_structured_output": final_state["needs_structured_output"],
                    "inference_result": final_state.get("inference_result"),
                    "structured_result": final_state.get("structured_result")
                },
                "final_response": final_state["final_response"],
                "response_type": final_state["response_type"],
                "execution_path": final_state["execution_path"],
                "errors": final_state["errors"],
                "langgraph_enabled": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "user_query": user_query,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }

# ================================
# MCP SERVER RUNNER (UNCHANGED)
# ================================

def run_mcp_server():
    """Run the FastMCP server as a separate process"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "mcp-server":
        task_mcp.run(transport="stdio")
    else:
        print("To run MCP server: python script.py mcp-server")

# ================================
# FASTAPI APPLICATION (UNCHANGED)
# ================================

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's query")
    user_id: str = Field(default="user_123", description="User identifier")

class QueryResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str

# Initialize FastAPI app
app = FastAPI(title="Dynamic Task Orchestrator with LangGraph", version="2.0.0")

# Global orchestrator instance
orchestrator = None

@app.on_event("startup")
async def startup_event():
    """Initialize orchestrator on startup"""
    global orchestrator
    # Replace with your OpenAI API key
    orchestrator = DynamicTaskOrchestrator(openai_api_key="your-openai-api-key")
    print("Dynamic Task Orchestrator with LangGraph started successfully")

@app.post("/api/query", response_model=QueryResponse)
async def process_query_endpoint(request: QueryRequest):
    """Main endpoint that uses LangGraph workflow"""
    try:
        result = await orchestrator.process_query(
            user_query=request.query,
            user_id=request.user_id
        )
        
        return QueryResponse(
            success=result["success"],
            data=result if result["success"] else None,
            error=result.get("error"),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        return QueryResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now().isoformat()
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "system_stats": context_manager.system_stats,
        "langgraph_enabled": True,
        "timestamp": datetime.now().isoformat()
    }

# ================================
# TESTING (SIMPLIFIED)
# ================================

async def test_langgraph_orchestrator():
    """Test the LangGraph enhanced orchestrator"""
    
    print("Testing Dynamic Task Orchestrator with LangGraph")
    print("=" * 60)
    
    orchestrator = DynamicTaskOrchestrator(openai_api_key="your-openai-api-key")
    
    test_queries = [
        {
            "query": "Show me my tasks",
            "expected": "Natural language response via LangGraph workflow"
        },
        {
            "query": "Generate a JSON report of my task status", 
            "expected": "Structured JSON via LangGraph workflow"
        },
        {
            "query": "What high priority tasks do I have?",
            "expected": "Natural language with filtering"
        },
        {
            "query": "Create a dashboard export",
            "expected": "Structured JSON output"
        }
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        expected = test_case["expected"]
        
        print(f"\nTest {i}: {query}")
        print(f"Expected: {expected}")
        print("-" * 50)
        
        try:
            result = await orchestrator.process_query(query, "user_123")
            
            if result["success"]:
                print(f"✅ Success")
                print(f"LangGraph enabled: {result['langgraph_enabled']}")
                print(f"Execution path: {' -> '.join(result['execution_path'])}")
                print(f"Response type: {result['response_type']}")
                print(f"Structured output needed: {result['workflow_results']['needs_structured_output']}")
                
                # Show response preview
                final_response = result['final_response']
                if result["response_type"] == "structured_json":
                    print("📊 Structured JSON Generated:")
                    print(str(final_response)[:200] + "...")
                else:
                    print(f"💬 Natural Language: {final_response[:150]}...")
                    
                if result["errors"]:
                    print(f"⚠️  Errors encountered: {result['errors']}")
                    
            else:
                print(f"❌ Error: {result['error']}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        print("=" * 60)

# ================================
# MAIN EXECUTION
# ================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "mcp-server":
        # Run as MCP server
        print("Starting FastMCP Task Server...")
        task_mcp.run(transport="stdio")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run tests
        asyncio.run(test_langgraph_orchestrator())
    
    elif len(sys.argv) > 1 and sys.argv[1] == "api":
        # Run FastAPI server
        print("Starting FastAPI server with LangGraph...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    
    else:
        print("""
Dynamic Task Orchestrator with LangGraph Usage:

Enhanced with LangGraph while keeping existing agents:
✅ Same Task Agent (fetches data from your API)
✅ Same Inference Agent (processes queries naturally)  
✅ Same Structured Output Agent (generates JSON)
✅ LangGraph orchestrates the workflow dynamically
✅ Memory and state management included
✅ Dynamic routing based on query type

Your Original Workflow (now with LangGraph):
1. Check if structured output needed
2. Task Agent fetches data  
3. Inference Agent processes query
4. Structured Output Agent (if needed)
5. Return appropriate response

Commands:
1. Run MCP Server: python script.py mcp-server
2. Run FastAPI: python script.py api  
3. Test: python script.py test

Benefits of LangGraph Enhancement:
- Dynamic routing and decision making
- Built-in state management and memory
- Error handling and retry logic
- Workflow visualization and debugging
- Easy to extend with new steps
""")

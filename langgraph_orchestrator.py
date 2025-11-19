from langgraph.graph import StateGraph, END
from typing import Any
import httpx
import asyncio
from datetime import datetime
import uuid

# Import the state definition (from previous artifact)
# from orchestrator_state import OrchestratorState, PlanStatus, StepStatus, etc.

class HierarchicalOrchestrator:
    """
    LangGraph-based Hierarchical Orchestrator for Agentic AI System
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the orchestrator workflow graph"""
        
        # Initialize graph with state
        workflow = StateGraph(OrchestratorState)
        
        # Add nodes for each phase
        workflow.add_node("understand_intent", self.understand_intent)
        workflow.add_node("discover_agents", self.discover_agents)
        workflow.add_node("create_plan", self.create_plan)
        workflow.add_node("validate_plan", self.validate_plan)
        workflow.add_node("execute_plan", self.execute_plan)
        workflow.add_node("execute_step", self.execute_step)
        workflow.add_node("aggregate_results", self.aggregate_results)
        workflow.add_node("handle_error", self.handle_error)
        workflow.add_node("request_user_input", self.request_user_input)
        
        # Define edges with conditional routing
        workflow.set_entry_point("understand_intent")
        
        workflow.add_conditional_edges(
            "understand_intent",
            self.route_after_intent,
            {
                "discover_agents": "discover_agents",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "discover_agents",
            self.route_after_discovery,
            {
                "create_plan": "create_plan",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "create_plan",
            self.route_after_planning,
            {
                "validate_plan": "validate_plan",
                "error": "handle_error",
                "user_input": "request_user_input"
            }
        )
        
        workflow.add_conditional_edges(
            "validate_plan",
            self.route_after_validation,
            {
                "execute_plan": "execute_plan",
                "create_plan": "create_plan",  # Replan if validation fails
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_plan",
            self.route_during_execution,
            {
                "execute_step": "execute_step",
                "aggregate_results": "aggregate_results",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_step",
            self.route_after_step,
            {
                "execute_plan": "execute_plan",  # Continue execution
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "aggregate_results",
            self.route_after_aggregation,
            {
                "end": END,
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "handle_error",
            self.route_after_error,
            {
                "create_plan": "create_plan",  # Replan
                "execute_plan": "execute_plan",  # Retry
                "user_input": "request_user_input",
                "end": END  # Abort
            }
        )
        
        workflow.add_conditional_edges(
            "request_user_input",
            self.route_after_user_input,
            {
                "create_plan": "create_plan",
                "execute_plan": "execute_plan",
                "end": END
            }
        )
        
        return workflow.compile()
    
    # ===== Node Implementations =====
    
    async def understand_intent(self, state: OrchestratorState) -> OrchestratorState:
        """Analyze user input and extract intent"""
        state["intent_understanding_status"] = "analyzing"
        state["updated_at"] = datetime.now()
        
        try:
            # Use LLM to understand intent
            intent_prompt = f"""
            Analyze the following user request and extract:
            1. Primary intent
            2. Sub-intents
            3. Entities mentioned
            4. Required capabilities
            
            User request: {state['current_user_input']}
            """
            
            # Call LLM (placeholder - integrate with your LLM)
            # response = await self.llm.ainvoke(intent_prompt)
            
            # Mock response for demonstration
            state["intent_analysis"] = {
                "intent_id": f"intent_{uuid.uuid4().hex[:8]}",
                "primary_intent": "extracted_intent",
                "sub_intents": [],
                "entities": {},
                "confidence": 0.9,
                "required_capabilities": ["capability1", "capability2"],
                "context": {}
            }
            
            state["intent_understanding_status"] = "completed"
            state["debug_logs"].append({
                "timestamp": datetime.now(),
                "node": "understand_intent",
                "message": "Intent analysis completed"
            })
            
        except Exception as e:
            state["intent_understanding_status"] = "failed"
            state["errors"].append({
                "error_id": f"err_{uuid.uuid4().hex[:8]}",
                "timestamp": datetime.now(),
                "step_id": None,
                "agent_id": None,
                "error_type": "IntentAnalysisError",
                "error_message": str(e),
                "stack_trace": None,
                "recovery_action": "retry",
                "is_recoverable": True
            })
        
        return state
    
    async def discover_agents(self, state: OrchestratorState) -> OrchestratorState:
        """Discover available agents and their capabilities"""
        state["agent_discovery_status"] = "discovering"
        state["updated_at"] = datetime.now()
        
        try:
            required_caps = state["intent_analysis"]["required_capabilities"]
            
            # Query agent registry
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{state['agent_registry_endpoint']}/discover",
                    json={"required_capabilities": required_caps},
                    timeout=10.0
                )
                agents = response.json()
            
            for agent in agents:
                state["discovered_agents"].append({
                    "agent_id": agent["id"],
                    "agent_name": agent["name"],
                    "endpoint": agent["endpoint"],
                    "capabilities": agent["capabilities"],
                    "status": agent["status"],
                    "avg_response_time": agent.get("avg_response_time"),
                    "success_rate": agent.get("success_rate"),
                    "metadata": agent.get("metadata", {})
                })
            
            state["agent_discovery_status"] = "completed"
            state["debug_logs"].append({
                "timestamp": datetime.now(),
                "node": "discover_agents",
                "message": f"Discovered {len(agents)} agents"
            })
            
        except Exception as e:
            state["agent_discovery_status"] = "failed"
            state["errors"].append({
                "error_id": f"err_{uuid.uuid4().hex[:8]}",
                "timestamp": datetime.now(),
                "step_id": None,
                "agent_id": None,
                "error_type": "AgentDiscoveryError",
                "error_message": str(e),
                "stack_trace": None,
                "recovery_action": "retry",
                "is_recoverable": True
            })
        
        return state
    
    async def create_plan(self, state: OrchestratorState) -> OrchestratorState:
        """Create execution plan with selected agents"""
        state["planning_status"] = "in_progress"
        state["planning_iterations"] += 1
        state["updated_at"] = datetime.now()
        
        try:
            intent = state["intent_analysis"]
            agents = state["discovered_agents"]
            
            # Use LLM to create plan
            planning_prompt = f"""
            Create an execution plan for the following:
            Intent: {intent['primary_intent']}
            Available agents: {[a['agent_name'] for a in agents]}
            
            Provide a step-by-step plan with:
            - Agent assignments
            - Execution order (sequential/parallel)
            - Dependencies between steps
            - Expected inputs/outputs
            """
            
            # Mock plan creation
            plan_id = f"plan_{uuid.uuid4().hex[:8]}"
            steps = []
            
            # Example: Create 2 steps
            for i, cap in enumerate(intent["required_capabilities"][:2]):
                step_id = f"step_{plan_id}_{i}"
                steps.append({
                    "step_id": step_id,
                    "step_number": i + 1,
                    "agent_id": agents[i % len(agents)]["agent_id"],
                    "agent_name": agents[i % len(agents)]["agent_name"],
                    "capability_id": cap,
                    "description": f"Execute {cap}",
                    "input_data": {},
                    "expected_output": None,
                    "dependencies": [steps[i-1]["step_id"]] if i > 0 else [],
                    "execution_mode": "sequential",
                    "status": "pending",
                    "retry_count": 0,
                    "max_retries": 3,
                    "started_at": None,
                    "completed_at": None,
                    "error": None
                })
            
            state["execution_plan"] = {
                "plan_id": plan_id,
                "created_at": datetime.now(),
                "steps": steps,
                "total_steps": len(steps),
                "execution_order": [[s["step_id"]] for s in steps],  # Sequential
                "status": "pending",
                "estimated_duration": None
            }
            
            state["pending_steps"] = [s["step_id"] for s in steps]
            state["planning_status"] = "completed"
            
        except Exception as e:
            state["planning_status"] = "failed"
            state["errors"].append({
                "error_id": f"err_{uuid.uuid4().hex[:8]}",
                "timestamp": datetime.now(),
                "step_id": None,
                "agent_id": None,
                "error_type": "PlanningError",
                "error_message": str(e),
                "stack_trace": None,
                "recovery_action": "retry",
                "is_recoverable": True
            })
        
        return state
    
    async def validate_plan(self, state: OrchestratorState) -> OrchestratorState:
        """Validate the execution plan"""
        # Check for circular dependencies, resource constraints, etc.
        plan = state["execution_plan"]
        
        # Simple validation: ensure all agents are available
        agent_ids = [s["agent_id"] for s in plan["steps"]]
        available_agents = [a["agent_id"] for a in state["discovered_agents"]]
        
        if all(aid in available_agents for aid in agent_ids):
            state["execution_plan"]["status"] = "validated"
        else:
            state["should_replan"] = True
        
        return state
    
    async def execute_plan(self, state: OrchestratorState) -> OrchestratorState:
        """Coordinate plan execution"""
        state["execution_status"] = PlanStatus.IN_PROGRESS
        state["updated_at"] = datetime.now()
        
        # Get next step(s) to execute
        pending = state["pending_steps"]
        
        if pending:
            # For simplicity, execute one step at a time
            state["current_step_id"] = pending[0]
        else:
            # All steps completed
            state["execution_status"] = PlanStatus.COMPLETED
        
        return state
    
    async def execute_step(self, state: OrchestratorState) -> OrchestratorState:
        """Execute a single step by invoking an agent"""
        step_id = state["current_step_id"]
        plan = state["execution_plan"]
        step = next(s for s in plan["steps"] if s["step_id"] == step_id)
        
        agent = next(a for a in state["discovered_agents"] 
                    if a["agent_id"] == step["agent_id"])
        
        invocation_id = f"inv_{uuid.uuid4().hex[:8]}"
        
        try:
            step["status"] = StepStatus.RUNNING
            step["started_at"] = datetime.now()
            
            # Prepare payload with thread_id for state persistence
            payload = {
                "thread_id": state["thread_id"],
                "session_id": state["session_id"],
                "input": step["input_data"],
                "context": state["shared_context"]
            }
            
            # Invoke agent
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{agent['endpoint']}/invoke",
                    json=payload,
                    timeout=30.0
                )
                result = response.json()
            
            # Record invocation
            invocation_record = {
                "invocation_id": invocation_id,
                "step_id": step_id,
                "agent_id": agent["agent_id"],
                "endpoint": agent["endpoint"],
                "input_payload": payload,
                "output_response": result,
                "thread_id": state["thread_id"],
                "invoked_at": step["started_at"],
                "completed_at": datetime.now(),
                "duration": (datetime.now() - step["started_at"]).total_seconds(),
                "status": StepStatus.COMPLETED,
                "error_details": None,
                "retry_attempt": step["retry_count"]
            }
            
            state["agent_invocations"].append(invocation_record)
            state["intermediate_results"][step_id] = result
            
            step["status"] = StepStatus.COMPLETED
            step["completed_at"] = datetime.now()
            
            state["completed_steps"].append(step_id)
            state["pending_steps"].remove(step_id)
            
        except Exception as e:
            step["status"] = StepStatus.FAILED
            step["error"] = {"message": str(e)}
            step["retry_count"] += 1
            
            state["failed_steps"].append(step_id)
            state["errors"].append({
                "error_id": f"err_{uuid.uuid4().hex[:8]}",
                "timestamp": datetime.now(),
                "step_id": step_id,
                "agent_id": agent["agent_id"],
                "error_type": "StepExecutionError",
                "error_message": str(e),
                "stack_trace": None,
                "recovery_action": "retry" if step["retry_count"] < step["max_retries"] else "abort",
                "is_recoverable": step["retry_count"] < step["max_retries"]
            })
        
        state["current_step_id"] = None
        return state
    
    async def aggregate_results(self, state: OrchestratorState) -> OrchestratorState:
        """Aggregate results from all steps"""
        results = state["intermediate_results"]
        
        # Combine results
        state["final_output"] = {
            "status": "success",
            "results": results,
            "summary": "Task completed successfully"
        }
        
        # Generate human-readable response
        state["aggregated_response"] = f"Completed {len(results)} steps successfully."
        
        return state
    
    async def handle_error(self, state: OrchestratorState) -> OrchestratorState:
        """Handle errors with recovery strategies"""
        state["error_recovery_attempts"] += 1
        
        last_error = state["errors"][-1] if state["errors"] else None
        
        if last_error and last_error["is_recoverable"]:
            if state["error_recovery_attempts"] < state["max_recovery_attempts"]:
                state["current_error_strategy"] = last_error["recovery_action"]
            else:
                state["should_abort"] = True
        else:
            state["should_abort"] = True
        
        return state
    
    async def request_user_input(self, state: OrchestratorState) -> OrchestratorState:
        """Request additional input from user"""
        state["requires_user_input"] = True
        return state
    
    # ===== Routing Functions =====
    
    def route_after_intent(self, state: OrchestratorState) -> str:
        if state["intent_understanding_status"] == "completed":
            return "discover_agents"
        return "error"
    
    def route_after_discovery(self, state: OrchestratorState) -> str:
        if state["agent_discovery_status"] == "completed":
            return "create_plan"
        return "error"
    
    def route_after_planning(self, state: OrchestratorState) -> str:
        if state["planning_status"] == "completed":
            return "validate_plan"
        elif state["requires_user_input"]:
            return "user_input"
        return "error"
    
    def route_after_validation(self, state: OrchestratorState) -> str:
        if state["should_replan"]:
            return "create_plan"
        elif state["execution_plan"]["status"] == "validated":
            return "execute_plan"
        return "error"
    
    def route_during_execution(self, state: OrchestratorState) -> str:
        if state["current_step_id"]:
            return "execute_step"
        elif state["execution_status"] == PlanStatus.COMPLETED:
            return "aggregate_results"
        return "error"
    
    def route_after_step(self, state: OrchestratorState) -> str:
        if state["errors"] and state["errors"][-1]["step_id"] == state.get("current_step_id"):
            return "error"
        return "execute_plan"
    
    def route_after_aggregation(self, state: OrchestratorState) -> str:
        if state["final_output"]:
            return "end"
        return "error"
    
    def route_after_error(self, state: OrchestratorState) -> str:
        if state["should_abort"]:
            return "end"
        elif state["requires_user_input"]:
            return "user_input"
        elif state["current_error_strategy"] == "retry":
            return "execute_plan"
        elif state["should_replan"]:
            return "create_plan"
        return "end"
    
    def route_after_user_input(self, state: OrchestratorState) -> str:
        # Logic based on user response
        return "create_plan"
    
    async def run(self, initial_state: OrchestratorState) -> OrchestratorState:
        """Execute the orchestrator workflow"""
        return await self.graph.ainvoke(initial_state)


# Example usage
async def main():
    config = {
        "agent_registry_endpoint": "http://registry.example.com",
        "max_recovery_attempts": 3
    }
    
    orchestrator = HierarchicalOrchestrator(config)
    
    initial_state = initialize_orchestrator_state(
        session_id="sess_123",
        thread_id="thread_456",
        user_input="Analyze sales data and generate report",
        config=config
    )
    
    final_state = await orchestrator.run(initial_state)
    print(f"Final output: {final_state['aggregated_response']}")

if __name__ == "__main__":
    asyncio.run(main())

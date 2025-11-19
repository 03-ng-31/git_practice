from typing import TypedDict, Annotated, Literal, Optional, Any
from datetime import datetime
from enum import Enum
import operator

# Enums for state management
class PlanStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_COMPLETED = "partially_completed"

class ExecutionMode(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRY = "retry"

# Sub-structures for organized state
class AgentCapability(TypedDict):
    """Represents capabilities of a discovered agent"""
    capability_id: str
    name: str
    description: str
    input_schema: dict
    output_schema: dict
    constraints: Optional[dict]

class DiscoveredAgent(TypedDict):
    """Information about an available agent"""
    agent_id: str
    agent_name: str
    endpoint: str
    capabilities: list[AgentCapability]
    status: Literal["available", "busy", "offline"]
    avg_response_time: Optional[float]
    success_rate: Optional[float]
    metadata: dict

class PlanStep(TypedDict):
    """Individual step in the execution plan"""
    step_id: str
    step_number: int
    agent_id: str
    agent_name: str
    capability_id: str
    description: str
    input_data: dict
    expected_output: Optional[dict]
    dependencies: list[str]  # step_ids that must complete first
    execution_mode: ExecutionMode
    status: StepStatus
    retry_count: int
    max_retries: int
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error: Optional[dict]

class ExecutionPlan(TypedDict):
    """Complete execution plan"""
    plan_id: str
    created_at: datetime
    steps: list[PlanStep]
    total_steps: int
    execution_order: list[list[str]]  # List of parallel execution groups
    status: PlanStatus
    estimated_duration: Optional[float]

class AgentInvocation(TypedDict):
    """Record of an agent invocation"""
    invocation_id: str
    step_id: str
    agent_id: str
    endpoint: str
    input_payload: dict
    output_response: Optional[dict]
    thread_id: str  # For state persistence
    invoked_at: datetime
    completed_at: Optional[datetime]
    duration: Optional[float]
    status: StepStatus
    error_details: Optional[dict]
    retry_attempt: int

class ErrorRecord(TypedDict):
    """Detailed error information"""
    error_id: str
    timestamp: datetime
    step_id: Optional[str]
    agent_id: Optional[str]
    error_type: str
    error_message: str
    stack_trace: Optional[str]
    recovery_action: Optional[str]
    is_recoverable: bool

class ConversationMessage(TypedDict):
    """User conversation history"""
    message_id: str
    timestamp: datetime
    role: Literal["user", "assistant", "system"]
    content: str
    metadata: Optional[dict]

class IntentAnalysis(TypedDict):
    """Analyzed user intent"""
    intent_id: str
    primary_intent: str
    sub_intents: list[str]
    entities: dict
    confidence: float
    required_capabilities: list[str]
    context: dict

# Main Orchestrator State
class OrchestratorState(TypedDict):
    """
    Comprehensive state for the Orchestrator in a Hierarchical Agentic System.
    Manages planning, execution, agent coordination, and error handling.
    """
    
    # Session Management
    session_id: str
    thread_id: str  # Persistent thread for state continuity
    created_at: datetime
    updated_at: datetime
    
    # User Interaction
    conversation_history: Annotated[list[ConversationMessage], operator.add]
    current_user_input: str
    user_context: dict  # User preferences, previous interactions, etc.
    
    # Intent Understanding
    intent_analysis: Optional[IntentAnalysis]
    intent_understanding_status: Literal["pending", "analyzing", "completed", "failed"]
    
    # Agent Discovery
    discovered_agents: Annotated[list[DiscoveredAgent], operator.add]
    agent_discovery_status: Literal["pending", "discovering", "completed", "failed"]
    agent_registry_endpoint: str
    
    # Planning Phase
    execution_plan: Optional[ExecutionPlan]
    planning_status: Literal["pending", "in_progress", "completed", "failed"]
    planning_iterations: int  # Track replanning attempts
    alternative_plans: list[ExecutionPlan]  # Backup plans
    
    # Execution Phase
    current_step_id: Optional[str]
    execution_status: PlanStatus
    completed_steps: Annotated[list[str], operator.add]  # step_ids
    failed_steps: Annotated[list[str], operator.add]  # step_ids
    pending_steps: list[str]  # step_ids
    
    # Agent Invocations
    agent_invocations: Annotated[list[AgentInvocation], operator.add]
    active_invocations: list[str]  # invocation_ids currently running
    
    # Inter-Agent Communication
    agent_handovers: Annotated[list[dict], operator.add]  # Track handover details
    shared_context: dict  # Context shared across agents
    
    # Results & Output
    intermediate_results: dict  # step_id -> result mapping
    final_output: Optional[dict]
    aggregated_response: Optional[str]  # Human-readable final response
    
    # Error Handling
    errors: Annotated[list[ErrorRecord], operator.add]
    error_recovery_attempts: int
    max_recovery_attempts: int
    current_error_strategy: Optional[str]
    
    # Monitoring & Observability
    performance_metrics: dict  # Execution times, agent response times, etc.
    debug_logs: Annotated[list[dict], operator.add]
    checkpoints: list[dict]  # State snapshots for rollback
    
    # Control Flow
    should_replan: bool
    should_abort: bool
    requires_user_input: bool
    next_action: Optional[str]  # Next node to execute in the graph
    
    # Configuration
    config: dict  # Orchestrator configuration (timeouts, retry policies, etc.)
    execution_constraints: dict  # Budget, time limits, resource constraints


# Helper function to initialize state
def initialize_orchestrator_state(
    session_id: str,
    thread_id: str,
    user_input: str,
    config: Optional[dict] = None
) -> OrchestratorState:
    """Initialize a new orchestrator state"""
    now = datetime.now()
    
    return OrchestratorState(
        # Session
        session_id=session_id,
        thread_id=thread_id,
        created_at=now,
        updated_at=now,
        
        # User Interaction
        conversation_history=[
            ConversationMessage(
                message_id=f"msg_{session_id}_0",
                timestamp=now,
                role="user",
                content=user_input,
                metadata={}
            )
        ],
        current_user_input=user_input,
        user_context={},
        
        # Intent
        intent_analysis=None,
        intent_understanding_status="pending",
        
        # Discovery
        discovered_agents=[],
        agent_discovery_status="pending",
        agent_registry_endpoint=config.get("agent_registry_endpoint", "") if config else "",
        
        # Planning
        execution_plan=None,
        planning_status="pending",
        planning_iterations=0,
        alternative_plans=[],
        
        # Execution
        current_step_id=None,
        execution_status=PlanStatus.PENDING,
        completed_steps=[],
        failed_steps=[],
        pending_steps=[],
        
        # Invocations
        agent_invocations=[],
        active_invocations=[],
        
        # Communication
        agent_handovers=[],
        shared_context={},
        
        # Results
        intermediate_results={},
        final_output=None,
        aggregated_response=None,
        
        # Errors
        errors=[],
        error_recovery_attempts=0,
        max_recovery_attempts=config.get("max_recovery_attempts", 3) if config else 3,
        current_error_strategy=None,
        
        # Monitoring
        performance_metrics={},
        debug_logs=[],
        checkpoints=[],
        
        # Control Flow
        should_replan=False,
        should_abort=False,
        requires_user_input=False,
        next_action="understand_intent",
        
        # Configuration
        config=config or {},
        execution_constraints={}
    )


# Example usage and state transitions
def example_state_flow():
    """
    Example showing how state evolves through orchestrator lifecycle
    """
    
    # 1. Initialize
    state = initialize_orchestrator_state(
        session_id="sess_123",
        thread_id="thread_456",
        user_input="Analyze the sales data and create a report",
        config={
            "agent_registry_endpoint": "http://registry.example.com",
            "max_recovery_attempts": 3
        }
    )
    
    # 2. After intent understanding
    state["intent_analysis"] = IntentAnalysis(
        intent_id="intent_789",
        primary_intent="data_analysis_and_reporting",
        sub_intents=["analyze_sales_data", "generate_report"],
        entities={"data_type": "sales", "output_format": "report"},
        confidence=0.95,
        required_capabilities=["data_analysis", "report_generation"],
        context={}
    )
    state["intent_understanding_status"] = "completed"
    
    # 3. After agent discovery
    state["discovered_agents"].append(
        DiscoveredAgent(
            agent_id="agent_analytics_001",
            agent_name="SalesAnalyticsAgent",
            endpoint="http://analytics.example.com/api",
            capabilities=[
                AgentCapability(
                    capability_id="cap_001",
                    name="data_analysis",
                    description="Analyzes sales data",
                    input_schema={"type": "object"},
                    output_schema={"type": "object"},
                    constraints={}
                )
            ],
            status="available",
            avg_response_time=2.5,
            success_rate=0.98,
            metadata={}
        )
    )
    state["agent_discovery_status"] = "completed"
    
    return state

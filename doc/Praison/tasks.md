Core Concepts
Tasks
Understanding Tasks in PraisonAI

​
Understanding Tasks
Tasks are units of work that agents execute. Each task has a clear goal, input requirements, and expected outputs.

Task
Role
Goal
Backstory
Expected Output
Input
Output
​
Core Components
Task Definition
Clear description of work to be done and expected outputs

Agent Assignment
Matching tasks with capable agents

Tool Access
Resources available for task completion

Output Handling
Managing and formatting task results

​
Task Configuration

Basic

Advanced

task = Task(
    description="Research AI trends",
    expected_output="Summary report",
    agent=research_agent
)
​
Understanding Tasks

⚙️ Execution
Processing & Tools
🤖 Agent Assignment
Matching & Delegation
📋 Task Definition
Description & Requirements
▶ Start
✓ Output


📥 Pending
⚡ In Progress
✅ Completed
❌ Failed
​
Task Types
1
Basic Task

Simple, single-operation tasks with clear inputs and outputs

2
Decision Task

Tasks involving choices and conditional paths

```python
decision_task = Task(
    type="decision",
    conditions={
        "success": ["next_task"],
        "failure": ["error_task"]
    }
)
```
3
Loop Task

Repetitive operations over collections

```python
loop_task = Task(
    type="loop",
    items=data_list,
    operation="process_item"
)
​
```

Task Relationships
Properly managing task dependencies is crucial for complex workflows. Always ensure proper context sharing and error handling.

​
Context Sharing
```python
task_a = Task(name="research")
task_b = Task(
    name="analyze",
    context=[task_a]  # Uses task_a's output
)
​
```

Task Dependencies
Relationship	Description	Example
Sequential	Tasks run one after another	Research → Analysis → Report
Parallel	Independent tasks run simultaneously	Data Collection + Processing
Conditional	Tasks depend on previous results	Success → Next Task, Failure → Retry
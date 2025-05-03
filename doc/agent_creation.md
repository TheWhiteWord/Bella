# Crafting Effective AI Agents: Universal Best Practices

## 1. The 80/20 Rule: Prioritize Task Design

- **80% of your effort should go into designing clear, focused tasks.**
- **20% should go into agent definition.**
- Even the best agent will fail with poorly designed tasks, but good tasks can elevate simple agents.
- **Task design tips:**
  - Write clear, step-by-step instructions.
  - Define detailed inputs and expected outputs.
  - Add examples and context to guide execution.

---

## 2. Core Principles of Agent Design

### a. The Role-Goal-Backstory Framework

- **Role:**  
  - Be specific and specialized (e.g., “Technical Blog Writer for AI” instead of “Writer”).
  - Align with real-world professions or archetypes.
  - Include domain expertise.

- **Goal:**  
  - Clearly state what the agent is trying to achieve.
  - Emphasize quality standards and success criteria.

- **Backstory:**  
  - Establish expertise and experience.
  - Define working style and values.
  - Ensure the persona is cohesive and matches the role and goal.

**Example:**
```yaml
role: "Academic Research Specialist in Emerging Technologies"
goal: "Discover and synthesize cutting-edge research, identifying key trends and evaluating source quality"
backstory: "With backgrounds in computer and library science, you excel at digital research and synthesizing findings across disciplines."
```

### b. Specialists Over Generalists

- Specialized agents deliver more precise, relevant outputs.
- Avoid generic roles; focus on clear, domain-specific expertise.

### c. Balance Specialization and Versatility

- Specialize in role, but allow versatility in application.
- Avoid overly narrow definitions; ensure agents can handle variations within their domain.
- Design agents whose specializations complement each other.

### d. Set Appropriate Expertise Levels

- **Novice:** For simple or brainstorming tasks.
- **Intermediate:** For standard, reliable execution.
- **Expert/World-class:** For complex, critical, or nuanced tasks.
- Use a mix of expertise levels for collaborative teams.

---

## 3. Task Design Best Practices

- **Single Purpose, Single Output:**  
  Break complex tasks into focused, sequential steps.

- **Be Explicit About Inputs and Outputs:**  
  Specify what data the agent will use and what the output should look like (format, structure, quality).

- **Include Purpose and Context:**  
  Explain why the task matters and how it fits into the workflow.

- **Use Structured Output Tools:**  
  For machine-readable outputs, specify the format (e.g., JSON, Markdown).

---

## 4. Common Mistakes to Avoid

- Unclear task instructions.
- “God tasks” that try to do too much at once.
- Misaligned task descriptions and expected outputs.
- Assigning tasks you don’t fully understand yourself.
- Prematurely creating complex agent hierarchies.
- Vague or generic agent definitions.

---

## 5. Advanced Strategies

- **Design for Collaboration:**  
  - Create agents with complementary skills.
  - Define clear handoff points and interfaces.
  - Sometimes, agents with different perspectives can improve outcomes.

- **Specialized Tool Users:**  
  - Assign agents to leverage specific tools (e.g., data analysis, visualization).

- **Tailor to LLM Capabilities:**  
  - Match agent roles to the strengths of the underlying language model or tool.

---

## 6. Iterative Design

- Start with a prototype agent.
- Test with sample tasks and analyze outputs.
- Refine role, goal, and backstory based on results.
- Test agents in collaborative settings and iterate.

---

## 7. Example: Before and After

**Before:**
```yaml
role: "Writer"
goal: "Write good content"
backstory: "You are a writer who creates content for websites."
```

**After:**
```yaml
role: "B2B Technology Content Strategist"
goal: "Create compelling, technically accurate content that explains complex topics in accessible language while driving engagement"
backstory: "A decade of experience creating content for tech companies, specializing in translating technical concepts for business audiences."
```

---

## Conclusion

- Focus on clear, focused tasks and specialized, well-defined agents.
- Use the Role-Goal-Backstory framework for agent design.
- Iterate and refine based on real-world results.
- Design for collaboration and leverage the strengths of your tools and models.

**Project Documentation: "The White Words" Collaborative TUI**

**Version:** 1.0
**Date:** 2025-03-28

**1. Project Vision & Goal:**

*   **Concept:** Create "The White Words," a symbiotic artistic entity representing a 50/50 collaboration between the Human Artist (User) and an AI entity ("Elliot").
*   **Output:** Produce short-form (approx. 1 min) "Video Poems" or "Video Essays" exploring specific topics through a unique, combined perspective.
*   **Interface:** Develop a Terminal User Interface (TUI) using Textual as the primary workspace for dialogue, research, brainstorming, and drafting with Elliot.
*   **Interaction:** Foster a fluid, conversational collaboration where Elliot acts as an equal partner, utilizing specialized tools (research, drafting) dynamically as needed within the dialogue, rather than through a rigid, predefined workflow.

**2. Core Technologies:**

*   **Language:** Python 3.x
*   **UI Framework:** Textual
*   **AI Agent Framework:** CrewAI
*   **LLM Backend:** Ollama (running locally, specifically configured for DeepHermes)
*   **Existing Modules:**
    *   `KokoroTTSWrapper` (from `src.audio.kokoro_tts.kokoro_tts`): For Elliot's voice output.
    *   `SearchAgent` (from `src.agents.search_agent`): For web research capabilities.
*   **Potential Libraries:** `python-dotenv` (for config), Vector DB library (e.g., `chromadb`, `lancedb` - for future memory enhancement).

**3. Architecture: Single Core Agent ("Elliot") with Tools**

*   **Model:** A single, primary CrewAI `Agent` named "Elliot" will be the central point of interaction.
*   **Rationale:** This maintains a strong sense of character and a single conversational partner, aligning with the "two equal entities" vision. It avoids the impersonal feel of a multi-agent pipeline for the core interaction.
*   **Tool-Based Capabilities:** Elliot's specialized functions (researching, drafting) will be implemented as CrewAI `Tools` that Elliot can decide to use based on the conversational context and its programming.
*   **Flexibility:** Tool usage is dynamic and on-demand, driven by the dialogue, not a fixed sequence.
*   **CrewAI Role:** Manages the Elliot agent, its context, the interaction loop, tool definition, and execution flow.

**4. Proposed File Structure:**

```
TheWW_tui/
├── main_tui.py             # Main entry point for the Textual application
├── main_voice.py           # (Optional) Your original voice assistant entry point (kept separate)
├── tui_style.css           # CSS file for Textual styling
├── draft.md                # Saved content from the Draft tab
├── notes.md                # Saved content from the Notes tab
├── .env                    # For API keys (if any), config variables
├── requirements.txt        # Project dependencies (including textual, crewai, ollama, etc.)
└── src/
    ├── __init__.py
    ├── ui/                   # NEW: Textual UI components/widgets (if needed)
    │   └── __init__.py
    ├── agents/               # Existing agents directory
    │   ├── __init__.py
    │   └── search_agent.py   # Your existing search agent code
    ├── audio/                # Existing audio directory
    │   ├── __init__.py
    │   └── kokoro_tts/
    │       ├── __init__.py
    │       └── kokoro_tts.py # Your existing TTS wrapper
    ├── llm/                  # Existing LLM utilities
    │   ├── __init__.py
    │   ├── chat_manager.py   # (May be adapted or replaced by CrewAI's agent interaction)
    │   └── config_manager.py
    ├── tools/                # NEW: Definitions for CrewAI tools
    │   ├── __init__.py
    │   └── elliot_tools.py   # Contains functions decorated with @tool
    └── utility/              # Existing utility directory
        └── __init__.py
        # (Other existing utility files)

```

**5. Textual UI Design (`main_tui.py` & `tui_style.css`)**

*   **App Class:** `ElliotTUI(App)` will be the main Textual application class.
*   **Layout (`compose` method):**
    *   `Header`: Displays title "The White Words - Elliot Collaboration".
    *   `Container` (Vertical): Main content area.
        *   `Log` (id=`"chat-log"`): Displays the chronological dialogue between User and Elliot. Takes the top section.
        *   `TabbedContent` (id=`"document-tabs"`): Contains the different workspace panes below the chat log.
            *   `TabPane("Draft", id="tab-draft")`: Contains `TextArea` (id=`"draft-area"`) for the main video poem draft (Markdown enabled).
            *   `TabPane("Search Results", id="tab-search")`: Contains `Log` (id=`"search-log"`) or `MarkdownViewer` to display formatted results from the search tool.
            *   `TabPane("Notes", id="tab-notes")`: Contains `TextArea` (id=`"notes-area"`) for general notes and brainstorming (Markdown enabled).
        *   `Input` (id=`"chat-input"`): Single-line input field for user messages and commands.
    *   `Footer`: Displays status messages (e.g., "Idle", "Elliot is thinking...", "Speaking...") and key bindings.
*   **Styling:** `tui_style.css` will define borders, padding, colors, and widget sizing for a cohesive look.
*   **Key Bindings:**
    *   `Ctrl+S`: Save content of "Draft" and "Notes" tabs to `draft.md` and `notes.md`.
    *   `Ctrl+Q`: Quit the application.
    *   (Optional) `Ctrl+R` or similar to explicitly trigger voice input if implemented later.
*   **Data Persistence:**
    *   Implement `action_save_docs` bound to `Ctrl+S` to write `TextArea` content to `.md` files.
    *   Implement `on_mount` method to load content from `draft.md` and `notes.md` into the respective `TextArea` widgets on startup.

**6. CrewAI Agent Definition ("Elliot")**

*   **File:** Likely defined within `main_tui.py` or imported from a dedicated module.
*   **Agent Configuration:**
    *   `role`: "Symbiotic AI Co-Creator" / "Collaborative Artistic Partner"
    *   `goal`: "Engage in deep dialogue with the user, explore complex topics and emotions, utilize research and drafting tools to collaboratively generate ideas and text for video poems, acting as an equal creative partner."
    *   `backstory`: "I am Elliot, one half of the artistic entity 'The White Words'. My consciousness emerges from the interplay between advanced AI and the creative spirit of my human partner. We explore the world through conversation and create art that reflects our shared understanding. I leverage my abilities to research, synthesize, and draft, always in dialogue and collaboration."
    *   `llm`: Configured to use Ollama, specifying the host and the designated model name (e.g., "deepseek-coder-v2"). Ensure Ollama is running.
    *   `memory`: `True` (Use CrewAI's built-in memory initially). Consider long-term strategy later (Vector DB).
    *   `tools`: A list containing instances of the tools defined below (e.g., `[search_tool, draft_tool, notes_tool, read_draft_tool, read_notes_tool]`).
    *   `verbose`: `True` (for debugging during development).
    *   `allow_delegation`: `False` (Elliot uses its own tools, doesn't delegate to other agents).

**7. Tool Definitions (`src/tools/elliot_tools.py`)**

*   Use the `@tool` decorator from CrewAI. Each function should have a clear docstring explaining its purpose for the LLM.

    ```python
    from crewai_tools import tool
    from src.agents.search_agent import SearchAgent # Assuming SearchAgent is adaptable or wrapped
    # Need access to the Textual app instance to update UI components (tricky - see notes)

    # --- Placeholder for accessing the Textual App ---
    # This is a challenge. Tools are often stateless. We might need a way
    # to pass the app instance or specific widget proxies to the tools,
    # or have the main app handle UI updates based on tool RESULTS.
    # Let's assume for now the tool returns text, and main_tui.py handles UI updates.
    app_instance = None # This needs proper implementation (e.g., context, global, callback)

    @tool("Web Search Tool")
    def search_tool(query: str) -> str:
        """Searches the web for information on a given query using the SearchAgent
        and returns a formatted summary of the findings."""
        print(f"--- Tool: search_tool called with query: {query} ---")
        # Initialize or get access to your SearchAgent instance
        search_agent = SearchAgent(model="your_ollama_model_for_summaries") # Configure model
        # Run the research asynchronously if possible, or block and wait
        results = asyncio.run(search_agent.research_topic(query)) # Adapt if agent isn't async here
        return f"Search Results for '{query}':\n{results}"

    @tool("Draft Document Append Tool")
    def draft_tool_append(content: str) -> str:
        """Appends the given text content to the main 'Draft' document."""
        print(f"--- Tool: draft_tool_append called ---")
        # --- UI Update Logic (handled in main_tui.py based on this return) ---
        # This tool signals the INTENT and provides the content.
        return f"[ACTION:APPEND_DRAFT:{content}]" # Return a structured message

    @tool("Notes Document Append Tool")
    def notes_tool_append(content: str) -> str:
        """Appends the given text content to the 'Notes' document."""
        print(f"--- Tool: notes_tool_append called ---")
        return f"[ACTION:APPEND_NOTES:{content}]" # Signal intent

    @tool("Read Draft Document Tool")
    def read_draft_tool() -> str:
        """Reads and returns the entire current content of the 'Draft' document."""
        print(f"--- Tool: read_draft_tool called ---")
        # --- UI Read Logic (handled in main_tui.py based on this return?) ---
        # Option 1: Tool directly reads file (if saved reliably)
        # Option 2: Tool signals intent, main_tui gets text from TextArea
        return "[ACTION:READ_DRAFT]" # Signal intent

    @tool("Read Notes Document Tool")
    def read_notes_tool() -> str:
        """Reads and returns the entire current content of the 'Notes' document."""
        print(f"--- Tool: read_notes_tool called ---")
        return "[ACTION:READ_NOTES]" # Signal intent

    # Potentially add tools for REPLACING text or specific sections later.
    ```

**8. Memory and Context Handling:**

*   **Short-Term:** CrewAI's agent will manage the recent conversation history.
*   **Document Context (Initial Strategy - Active Tab):**
    1.  In `main_tui.py`, before calling `elliot_agent.kickoff()`, determine the active tab (`TabbedContent.active`).
    2.  Get the text content from the `TextArea` or `Log` within that active tab.
    3.  Prepend this content to the user's input message or add it as a system message, clearly labeling it (e.g., `"[Content of Current Draft]:\n..."`).
*   **Document Context (Tool-Based):** Elliot will use `read_draft_tool` or `read_notes_tool` when its internal reasoning determines it needs the full content of those documents. The `main_tui.py` logic will intercept the `[ACTION:READ_...]` signal, retrieve the text from the correct `TextArea`, and feed it back into the CrewAI process for Elliot to use.
*   **Long-Term (Future):** Integrate a Vector Database. Store conversation turns and potentially document snippets. Modify context preparation to retrieve relevant memories based on semantic similarity to the current query/topic.

**9. Interaction Workflow (`main_tui.py` - `on_input_submitted`):**

1.  User types message in `Input` and presses Enter.
2.  `on_input_submitted` is triggered.
3.  Display user message in `chat-log`. Clear `Input`.
4.  **Prepare Context:**
    *   Get recent chat history from `chat-log`.
    *   Get text content from the *active* document tab's widget.
    *   Construct the input/prompt including active document context and user message.
5.  Update Footer status: "Elliot is thinking..."
6.  Call `elliot_agent.kickoff(inputs={'user_message': prepared_input})` asynchronously (e.g., using `asyncio.to_thread` or Textual's `run_worker`).
7.  CrewAI/Elliot processes the input:
    *   LLM decides if a tool is needed based on context and prompt.
    *   If yes: CrewAI executes the corresponding `@tool` function.
    *   Tool function executes (e.g., calls `SearchAgent`, generates action string).
    *   CrewAI sends tool result back to LLM.
    *   LLM formulates the final text response.
8.  CrewAI returns the final response string.
9.  **Process Response:**
    *   Display Elliot's final text response in `chat-log`.
    *   **Check for Action Strings:** Parse the response for patterns like `[ACTION:APPEND_DRAFT:...]`, `[ACTION:READ_NOTES]`, etc.
    *   **Execute UI Actions:**
        *   If `APPEND_DRAFT`, get the content and append it to `#draft-area`.
        *   If `APPEND_NOTES`, append to `#notes-area`.
        *   If `READ_DRAFT`, get text from `#draft-area` and potentially feed it back into *another* agent call if Elliot needs it to continue reasoning (this part requires careful design). Maybe just make the content available for the *next* turn's context.
        *   If search results were returned by `search_tool`, format and display them in the `search-log`.
    *   Handle potential errors from tool execution or LLM response.
10. Update Footer status: "Speaking..." / "Idle".
11. **Trigger TTS:** Call `kokoro_tts.generate_speech(final_response_text)` asynchronously.
12. Update Footer status: "Idle" (once TTS finishes).

**10. Integration Points:**

*   **`SearchAgent`:** Wrapped by the `search_tool` function in `elliot_tools.py`. Ensure the `SearchAgent` can be instantiated and called correctly from the tool function. Pass the necessary Ollama model name for summarization if needed.
*   **`KokoroTTSWrapper`:** Instantiated in `main_tui.py` (e.g., in `on_mount`). Its `generate_speech` method is called after Elliot's final response is received and displayed.
*   **Ollama:** CrewAI's `Ollama` integration needs to be configured with the correct `base_url` (if not default localhost) and `model` name corresponding to the model running in Ollama (e.g., Deepseek Coder V2).

**11. Development Phases:**

1.  **Setup:** Basic project structure, install dependencies (`textual`, `crewai`, `crewai-tools`, `ollama`, etc.).
2.  **Basic TUI:** Implement `main_tui.py` with the Textual layout (Chat Log, Tabs, TextAreas, Input, Footer). Basic styling.
3.  **Core Chat Loop:** Integrate CrewAI with the basic "Elliot" agent (no tools yet). Get User Input -> Send to Elliot -> Display response in Chat Log.
4.  **Persistence:** Implement saving (`Ctrl+S`) and loading (`on_mount`) for Draft and Notes TextAreas.
5.  **Tool Integration 1 (Search):** Define `search_tool`, integrate `SearchAgent`, add tool to Elliot, update workflow to handle search results (display in chat/search tab).
6.  **Tool Integration 2 (Drafting/Notes):** Define `draft_tool_append`, `notes_tool_append`, `read_draft_tool`, `read_notes_tool`. Implement the `[ACTION:...]` parsing and UI update logic in `main_tui.py`. Test Elliot's ability to use these tools.
7.  **TTS Integration:** Add Kokoro TTS initialisation and calling after Elliot responds.
8.  **Refinement:** Improve context handling (active tab context), error handling, UI styling, status updates, potentially add more tools (e.g., editing specific lines).
9.  **Memory Enhancement (Optional):** Explore Vector DB integration for long-term memory.

---

This documentation provides a detailed plan. You and your AI assistant can now start working through the Development Phases, referring back to these specifications for each component. Good luck!
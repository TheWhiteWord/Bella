# Bella Functions

righ now BElla has these functions that are used to manage and interact with memory. These functions are designed to help Bella remember important information, recall memories, and manage the memory database:

- Use 'semantic_memory_search' when the user asks you to recall information from your memory.
        - Use 'save_to_memory' when the user shares important information they want you to remember.
        - Use 'read_specific_memory' when you need to access a specific memory by ID.
        - Use 'save_conversation' when the conversation contains valuable information worth saving.
        - Use 'evaluate_memory_importance' to determine if information is worth remembering.
        - Use 'list_memories_by_type' when you want to see what memories are available.

However the automatic memory managment and Bella function calling are creating a lot of noise in the conversation, and there is issues with duplicate memories being created, and vector store retrival, as i think the format stored in the md files by bella and by the automatic system are different.
Note that there are no existing data taht need to be migrated, as this is a new system and the old one was not used in production.

So after analyzing the situation i decided to modify the Bella functions to become a system for us to work on ideas and projects.
- The new system for memory tools that Bella has, will be more like a project management system, where we can create, edit, and delete projects. These will be more like the old system, where the user can call the functions directly, and the memory is stored in a more structured way. 

- The new system for Bella function calling will have a more structured way of storing memories, with folders and files for each project:


- We will have a new 'projects' folder.
- The project folder will have a list of projects, each with its own folder, each named on creation by a fuction call that will respond to the user request to "create a new folder" which will include a subject, like "create a new project, we will discuss love and war", and the assistant will create a folder named "love and war" in the projects folder, and all the standard subfolders necessary for the project.
  - the subfolders that are created on the project folder will be:
    - subfolder "notes":
    - subfolder "concepts"
    - subfolder "research"
    - subfolder "content"
The projects folder will be directly accessible for writing and retreiving by Bella function callings, but accessible by the autonomopus memory layer ONLY for reading and retreiving:
that means that bella can read and write to the project folder, but the memory layer can only read from it, and not write to it.

During the conversation, the user can call the function to:

---
# FUNCTIONS
---

- 'Start_project': where he assistant will create a new folder in the projects folder, named after the project and automatically all the file necessary for the project.
    - The assistant will create a new file in the project folder, named "notes".
    - The assistant will create a new file in the project folder, named "instructions".
    - The assistant will create a new file in the project folder, named "research".
    - The assistant will create a new file in the project folder, named "main".

NOTE: If the project already exists, the assistant will not create a new folders, but will use the existing one. 
While a project is active, the assistant will be able to read and write in and from the project folder.
The project will be active untill 'Quit_project' function (see at the end) is called.

    - Command: "Lets start the/a/a new project titled (project name)"
    
    - Full Example:
        - "Lets start a new project titled Love and War"
      - Result:
        - Project folder created: "Love and War"
        - Subfolders created:
          - notes
          - concepts
          - research
          - main
        - The assistant will be able to read and write to the project folder.
 
---

- 'Save_to': where the assistant can add a new entry to the notes file, or the research file, or the instructions file.

  - Command: "please, save to (file name) : \"message to be saved here\""
  
  - Examples:
    - "please, save to notes : \"message to be saved here\""
    - "please, save to research : \"message to be saved here\""
    - "please, save to instructions : \"message to be saved here\""
    - "please, save to main : \"message to be saved here\""
  
  NOTE: if the (file name) is not specified, the assistant will save to the notes file by default.

---

- 'Save_conversation': where the assistant can save an entry based on the context of the content from the last two messages (two user messages and two assistant replies) from the conversation to the notes file, or the research file, or the instructions file. Main is excluded from this function.
  
  - Command: "please, save conversation to (file name)"
  
  - Examples:
    - "please, save conversation to notes "
    - "please, save conversation to research "
    - "please, save conversation to instructions "
  - if the (file name) is not specified, the assistant will save to the notes file by default.
    
  - Full example:
    Interaction:
      - User: "what do you think about love and war?"
      - Assistant: "I think love is a powerful force that can conquer all obstacles, including war. It is a powerful emotion that can bring people together and create peace."
      - User: "don't you think that love can also be a weapon of war?"
      - Assistant: "Yes, love can be a powerful weapon in war. It can be used to manipulate and control people, and it can lead to violence and conflict."
  
    - Command: "please, save the conversation to the notes "
    
    - Result:
      - Title:"relation of love and war"
      - Content: "Love is the ultimate force that can conquer all obstacles, including war. It is a powerful emotion that can bring people together and create peace. However, love can also be a weapon of war, and it can be used to manipulate and control people. It can be a tool for destruction, and it can lead to violence and conflict."
      - Timestamp: "2023-10-01 12:00:00"
      - Project: "Love and War"
      - File: "notes"
      - Tag: "love, war, relationship"
      - Entry ID: "1234567890"


---

- 'Edit': where the assistant can edit a specific entry in the notes file, or the research file, or the instructions file. Since main is only one entry, the command will refer to is only as main.
  - Command: "please, edit (entry title or file name) in (file name): \"message to be edited here\""
  - Examples:
    - "please, edit entry \"entry title\" in notes: \"context to drive edit here\""
    - "please, edit entry \"entry title\" in research: \"context to drive edit here\""
    - "please, edit entry \"entry title\" in instructions: \"context to drive edit here\""
    - "please, edit main: \"context to drive edit here\""

    
  - Full example:
    - State:
      - Title:"love defeats war"
      - Content: "Love is the ultimate force that can conquer all obstacles, including war. It is a powerful emotion that can bring people together and create peace."
      - Timestamp: "2023-10-01 12:00:00"
      - Project: "Love and War"
      - File: "notes"
      - Tag: "love, war, relationship"
      - Entry ID: "1234567890"
  
    - Command: "please, edit entry love ideal : love can also be a weapon of war, and it can be used to manipulate and control people. It can be a tool for destruction, and it can lead to violence and conflict." 
    
    - Result:
      - Title:"love defeats war"
      - Content: "Love is the ultimate force that can conquer all obstacles, including war. It is a powerful emotion that can bring people together and create peace. However, love can also be a weapon of war, and it can be used to manipulate and control people. It can be a tool for destruction, and it can lead to violence and conflict."
      - Timestamp: "2023-10-01 12:03:00" #timestamp changes based on last edit
      - Project: "Love and War"
      - File: "notes"
      - Tag: "love, war, relationship"
      - Entry ID: "1234567890"

The assistant will search for the entry title in the specified file, and if it finds it, it will edit the entry with the context of the new message.

---

- 'List_all': where the assistant can list all entries titles from the notes file, the research file, or the instructions file or all notes related to a specific query. Main is excluded from this function.

  - Command: "please, list all (file name) in (project name)"
    - Examples:
    Unfiltered:
      - "please, list all notes "
      - "please, list all research "
      - "please, list all instructions "
    returns:
      - In (file name) we have (list of entries titles in specific file)
    
    Filtered:
      - "please, list all notes  related to violence"
      - "please, list all research  related to Plato"
      - "please, list all instructions  related to style"
    returns:
      - In (file name) we have (list of entries titles in specific file filetered by query)

If no filiter is specified, the assistant will list all entries in the notes file by default.

  - Full example:
    - State
      - Project title: 'Love and War'
      - entries titles in the file 'notes': "love defeats war", "war is evil ", "love ideal"
      - entries titles in the file 'research': "Andy Warrolol on love", "sophistic love", "Plato and love"
      - entries titles in the file 'instructions': "writing style", "good content", "format"
    
    (Unfiltered)
    - command: "please, list all notes "
      - Result:
        - In notes we have: "love defeats war", "war is evil ", "love ideal"
    - command: "please, list all research "
      - Result:
        - In research we have: "Andy Warrolol on love", "sophistic love", "Plato and love"
    -  command: "please, list all instructions "
      - Result:
        - In instructions we have: "writing style", "good content", "format"
    
    (Filtered)
    - command: "please, list all notes  related to violence"
      - Result:
        - In notes we have: "war is evil ", "love ideal"
    - command: "please, list all research  related to Plato"
      - Result:
        - In research we have: "Plato and love"
    - command: "please, list all instructions  related to style"
      - Result:
        - In instructions we have: "writing style"
    
---

- 'Read entry': where the assistant can read a specific entry in the notes file, or the research file, or the instructions file. Since main is only one entry, the command will refer to it only as main.
    
    - Command: "please, read entry (entry title)"
    - Examples:
        - "please, read entry \"entry title\" "
        returns:
        - "entry title": "content of entry"

    - Full example:
    - State
      - Title: "love defeats war"
      - Content: "Love is the ultimate force that can conquer all obstacles, including war. It is a powerful emotion that can bring people together and create peace."
      - Timestamp: "2023-10-01 12:00:00"
      - Project: "Love and War"
      - File: "notes"
      - Tag: "love, war, relationship"
      - Entry ID: "1234567890"

      - Title: "main"
      - Content: "Oh love, you that are so good yet so bad, you are the best and the worst, you are the light and the dark, you are the good and the evil, you are the love and the war."
      - Timestamp: "2023-10-01 12:00:00"
      - Project: "Love and War"
      - File: "main"
      - Tag: "poem, love, war"
      - Entry ID: "1232837890"    
  
    - command: "please, read entry love defeats war "
      - Result:
        - love defeats war: "Love is the ultimate force that can conquer all obstacles, including war. It is a powerful emotion that can bring people together and create peace."
    - command: "please, read main "
      - Result:
        - "Oh love, you that are so good yet so bad, you are the best and the worst, you are the light and the dark, you are the good and the evil, you are the love and the war."
  
---

- 'Delete': where the assistant can delete a specific entry in the notes file, or the research file, or the instructions file.
    - Command: "please, delete entry (entry title)"
    - Examples:
        - "please, delete entry "entry title" "


---

- Quit_project: this command serves to end the session on a specific project in the assistant's context and save any necessary data before exiting.
    - Command: "please, quit project (project name)"
    - Examples:
        - "please, quit project Love and War"
    
    - Result:
        - The assistant will save any necessary data and close the project.
        - The assistant will not be able to read or write with her function calling to the project folder until the project is reopened.
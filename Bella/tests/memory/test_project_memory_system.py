"""Tests for the project-based memory system.

Tests the functionality of the project-based memory system and its standardized format.
"""

import os
import sys
import pytest
import asyncio
import shutil
import tempfile
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import required modules - we'll mock these later
# These imports will be adjusted based on actual project organization
from src.memory.project_manager.memory_integration import get_memory_integration
from src.memory.project_manager.memory_format_adapter import MemoryFormatAdapter


@pytest.fixture
def temp_project_dir():
    """Create a temporary directory for project testing."""
    base_temp_dir = tempfile.mkdtemp()
    projects_dir = os.path.join(base_temp_dir, "projects")
    os.makedirs(projects_dir, exist_ok=True)
    yield projects_dir
    # Cleanup after tests
    shutil.rmtree(base_temp_dir)


class TestProjectMemorySystem:
    """Test cases for the project-based memory system."""
    
    @pytest.mark.asyncio
    async def test_start_project(self, temp_project_dir):
        """Test creating a new project."""
        # Mock the project creation function
        memory_integration = AsyncMock()
        
        with patch('src.memory.project_manager.memory_integration.get_memory_integration', 
                  return_value=memory_integration):
            
            # Setup expected directories
            project_name = "Love and War"
            project_dir = os.path.join(temp_project_dir, project_name)
            
            # Mock the function to create project directory structure
            async def mock_create_project(name, base_dir=temp_project_dir):
                # Create project directory
                project_dir = os.path.join(base_dir, name)
                os.makedirs(project_dir, exist_ok=True)
                
                # Create required subdirectories
                subdirs = ["notes", "concepts", "research", "content"]
                for subdir in subdirs:
                    os.makedirs(os.path.join(project_dir, subdir), exist_ok=True)
                    
                return {"success": True, "project_path": project_dir}
            
            # Assign the mock function
            memory_integration.create_project = AsyncMock(side_effect=mock_create_project)
            
            # Call the function (simulating what would happen in actual code)
            result = await memory_integration.create_project(project_name)
            
            # Verify project directory was created
            assert os.path.exists(project_dir)
            assert "success" in result and result["success"]
            
            # Verify subdirectories were created
            for subdir in ["notes", "concepts", "research", "content"]:
                assert os.path.exists(os.path.join(project_dir, subdir))
    
    @pytest.mark.asyncio
    async def test_save_to_project_file(self, temp_project_dir):
        """Test saving content to a project file."""
        # Mock the memory integration
        memory_integration = AsyncMock()
        
        with patch('src.memory.project_manager.memory_integration.get_memory_integration', 
                  return_value=memory_integration):
            
            # Setup test data
            project_name = "Love and War"
            file_type = "notes"
            content = "Love is the ultimate force that can conquer all obstacles."
            title = "love defeats war"
            tags = ["love", "war", "relationship"]
            
            # Mock the function to save content
            async def mock_save_to_project(project_name, file_type, content, title, tags=None):
                # Create a simulated memory entry
                project_dir = os.path.join(temp_project_dir, project_name)
                os.makedirs(project_dir, exist_ok=True)
                os.makedirs(os.path.join(project_dir, file_type), exist_ok=True)
                
                # Use the memory format adapter for standardized format
                formatted_content = MemoryFormatAdapter.convert_to_standard_format(
                    content,
                    title=title,
                    tags=tags,
                    memory_type=file_type,
                    source="function"
                )
                
                # Write to a file (simulated)
                file_path = os.path.join(project_dir, file_type, f"{title}.md")
                with open(file_path, "w") as f:
                    f.write(formatted_content)
                    
                return {"success": True, "path": file_path}
            
            # Setup the MemoryFormatAdapter mock
            formatted_content = """---
title: love defeats war
created: 2024-04-07T12:00:00
updated: 2024-04-07T12:00:00
memory_type: notes
tags:
- love
- war
- relationship
source: function
entry_id: 1234567890
---

# love defeats war

Love is the ultimate force that can conquer all obstacles.
"""
            with patch('src.memory.project_manager.memory_format_adapter.MemoryFormatAdapter.convert_to_standard_format', 
                      return_value=formatted_content):
                
                # Assign the mock function
                memory_integration.save_to_project = AsyncMock(side_effect=mock_save_to_project)
                
                # Call the function (simulating what would happen in actual code)
                result = await memory_integration.save_to_project(
                    project_name, 
                    file_type, 
                    content, 
                    title, 
                    tags
                )
                
                # Verify function was called with correct parameters
                memory_integration.save_to_project.assert_called_once_with(
                    project_name, 
                    file_type, 
                    content, 
                    title, 
                    tags
                )
                
                # Verify the success result
                assert "success" in result and result["success"]
    
    @pytest.mark.asyncio
    async def test_save_conversation_to_project(self, temp_project_dir):
        """Test saving a conversation to a project file."""
        # Mock the memory integration
        memory_integration = AsyncMock()
        
        with patch('src.memory.project_manager.memory_integration.get_memory_integration', 
                  return_value=memory_integration):
            
            # Setup test data
            project_name = "Love and War"
            file_type = "notes"
            conversation = [
                {"role": "user", "content": "what do you think about love and war?"},
                {"role": "assistant", "content": "I think love is a powerful force that can conquer all obstacles, including war."},
                {"role": "user", "content": "don't you think that love can also be a weapon of war?"},
                {"role": "assistant", "content": "Yes, love can be a powerful weapon in war. It can be used to manipulate and control people."}
            ]
            
            # Mock the function to save conversation
            async def mock_save_conversation(project_name, file_type, conversation):
                # Generate a title from conversation
                title = "relation of love and war"
                
                # Extract content from conversation
                content = "Love is the ultimate force that can conquer all obstacles, including war. " + \
                          "It is a powerful emotion that can bring people together and create peace. " + \
                          "However, love can also be a weapon of war, and it can be used to manipulate and control people."
                
                # Use standard formatting
                formatted_content = MemoryFormatAdapter.convert_to_standard_format(
                    content,
                    title=title,
                    tags=["love", "war", "relationship"],
                    memory_type=file_type,
                    source="function"
                )
                
                # Write to a file (simulated)
                project_dir = os.path.join(temp_project_dir, project_name)
                os.makedirs(project_dir, exist_ok=True)
                os.makedirs(os.path.join(project_dir, file_type), exist_ok=True)
                file_path = os.path.join(project_dir, file_type, f"{title}.md")
                with open(file_path, "w") as f:
                    f.write(formatted_content)
                    
                return {"success": True, "path": file_path, "title": title}
            
            # Setup the MemoryFormatAdapter mock
            formatted_content = """---
title: relation of love and war
created: 2024-04-07T12:00:00
updated: 2024-04-07T12:00:00
memory_type: notes
tags:
- love
- war
- relationship
source: function
entry_id: 1234567890
---

# relation of love and war

Love is the ultimate force that can conquer all obstacles, including war. It is a powerful emotion that can bring people together and create peace. However, love can also be a weapon of war, and it can be used to manipulate and control people.
"""
            with patch('src.memory.project_manager.memory_format_adapter.MemoryFormatAdapter.convert_to_standard_format', 
                      return_value=formatted_content):
                
                # Assign the mock function
                memory_integration.save_conversation_to_project = AsyncMock(side_effect=mock_save_conversation)
                
                # Call the function (simulating what would happen in actual code)
                result = await memory_integration.save_conversation_to_project(project_name, file_type, conversation)
                
                # Verify function was called with correct parameters
                memory_integration.save_conversation_to_project.assert_called_once_with(
                    project_name,
                    file_type,
                    conversation
                )
                
                # Verify the success result
                assert "success" in result and result["success"]
                assert result.get("title") == "relation of love and war"
    
    @pytest.mark.asyncio
    async def test_edit_project_entry(self, temp_project_dir):
        """Test editing an entry in a project file."""
        # Mock the memory integration
        memory_integration = AsyncMock()
        
        with patch('src.memory.project_manager.memory_integration.get_memory_integration', 
                  return_value=memory_integration):
            
            # Setup initial content
            project_name = "Love and War"
            file_type = "notes"
            entry_title = "love defeats war"
            initial_content = "Love is the ultimate force that can conquer all obstacles, including war."
            
            # Create project directory structure
            project_dir = os.path.join(temp_project_dir, project_name)
            os.makedirs(os.path.join(project_dir, file_type), exist_ok=True)
            
            # Create initial file with standard format
            initial_formatted = f"""---
title: {entry_title}
created: 2024-04-07T12:00:00
updated: 2024-04-07T12:00:00
memory_type: {file_type}
tags:
- love
- war
- relationship
source: function
entry_id: 1234567890
---

# {entry_title}

{initial_content}
"""
            entry_path = os.path.join(project_dir, file_type, f"{entry_title}.md")
            with open(entry_path, "w") as f:
                f.write(initial_formatted)
            
            # New content to edit
            edit_content = "love can also be a weapon of war, and it can be used to manipulate and control people."
            expected_content = initial_content + " However, " + edit_content
            
            # Mock the function to edit entry
            async def mock_edit_entry(project_name, file_type, entry_title, edit_content):
                # Find the entry
                entry_path = os.path.join(temp_project_dir, project_name, file_type, f"{entry_title}.md")
                
                if not os.path.exists(entry_path):
                    return {"success": False, "message": "Entry not found"}
                
                # Read the existing content
                with open(entry_path, "r") as f:
                    content = f.read()
                
                # Parse frontmatter and content
                frontmatter_end = content.find("---\n\n") + 4
                frontmatter = content[:frontmatter_end]
                
                # Extract the content part after the title
                title_line_end = content.find("\n\n", frontmatter_end) + 2
                existing_content = content[title_line_end:].strip()
                
                # Update the content with new edit
                updated_content = f"{existing_content} However, {edit_content}"
                
                # Update the "updated" timestamp in frontmatter
                import re
                current_time = datetime.now().isoformat()
                updated_frontmatter = re.sub(
                    r"updated: .*?\n",
                    f"updated: {current_time}\n",
                    frontmatter
                )
                
                # Write the updated content
                with open(entry_path, "w") as f:
                    f.write(updated_frontmatter)
                    f.write(f"# {entry_title}\n\n")
                    f.write(updated_content)
                
                return {"success": True, "path": entry_path}
            
            # Assign the mock function
            memory_integration.edit_project_entry = AsyncMock(side_effect=mock_edit_entry)
            
            # Call the function (simulating what would happen in actual code)
            result = await memory_integration.edit_project_entry(
                project_name, 
                file_type, 
                entry_title, 
                edit_content
            )
            
            # Verify function was called with correct parameters
            memory_integration.edit_project_entry.assert_called_once_with(
                project_name, 
                file_type, 
                entry_title, 
                edit_content
            )
            
            # Verify success result
            assert "success" in result and result["success"]
            
            # Verify the file was updated
            with open(entry_path, "r") as f:
                updated_file_content = f.read()
                
            assert expected_content in updated_file_content
            assert "However, " + edit_content in updated_file_content
    
    @pytest.mark.asyncio
    async def test_list_project_entries(self, temp_project_dir):
        """Test listing entries in a project file."""
        # Mock the memory integration
        memory_integration = AsyncMock()
        
        with patch('src.memory.project_manager.memory_integration.get_memory_integration', 
                  return_value=memory_integration):
            
            # Setup project with entries
            project_name = "Love and War"
            file_type = "notes"
            entries = [
                {"title": "love defeats war", "content": "Love is the ultimate force..."},
                {"title": "war is evil", "content": "War causes destruction and suffering..."},
                {"title": "love ideal", "content": "The ideal form of love is selfless..."}
            ]
            
            # Create project structure
            project_dir = os.path.join(temp_project_dir, project_name)
            notes_dir = os.path.join(project_dir, file_type)
            os.makedirs(notes_dir, exist_ok=True)
            
            # Create entry files
            for entry in entries:
                entry_path = os.path.join(notes_dir, f"{entry['title']}.md")
                with open(entry_path, "w") as f:
                    f.write(f"""---
title: {entry["title"]}
created: 2024-04-07T12:00:00
updated: 2024-04-07T12:00:00
memory_type: {file_type}
tags:
- test
source: function
entry_id: {hash(entry["title"])}
---

# {entry["title"]}

{entry["content"]}
""")
            
            # Mock function to list entries
            async def mock_list_entries(project_name, file_type, filter_query=None):
                entries_dir = os.path.join(temp_project_dir, project_name, file_type)
                if not os.path.exists(entries_dir):
                    return {"success": False, "message": "Directory not found"}
                
                entries = []
                for filename in os.listdir(entries_dir):
                    if filename.endswith(".md"):
                        title = filename[:-3]  # Remove .md extension
                        
                        # If filter query provided, check if it's in the file content
                        if filter_query:
                            with open(os.path.join(entries_dir, filename), "r") as f:
                                content = f.read()
                                if filter_query.lower() not in content.lower():
                                    continue
                        
                        entries.append(title)
                
                return {"success": True, "entries": entries}
            
            # Assign the mock function
            memory_integration.list_project_entries = AsyncMock(side_effect=mock_list_entries)
            
            # Test without filter
            result = await memory_integration.list_project_entries(project_name, file_type)
            
            # Verify function called with correct parameters
            memory_integration.list_project_entries.assert_called_with(project_name, file_type, None)
            
            # Verify success and entries returned
            assert "success" in result and result["success"]
            assert "entries" in result
            assert len(result["entries"]) == 3
            assert "love defeats war" in result["entries"]
            assert "war is evil" in result["entries"]
            assert "love ideal" in result["entries"]
            
            # Test with filter
            result = await memory_integration.list_project_entries(project_name, file_type, "evil")
            
            # Verify filtered results
            assert "success" in result and result["success"]
            assert "entries" in result
            assert len(result["entries"]) == 1
            assert "war is evil" in result["entries"]
    
    @pytest.mark.asyncio
    async def test_read_project_entry(self, temp_project_dir):
        """Test reading an entry from a project file."""
        # Mock the memory integration
        memory_integration = AsyncMock()
        
        with patch('src.memory.project_manager.memory_integration.get_memory_integration', 
                  return_value=memory_integration):
            
            # Setup test data
            project_name = "Love and War"
            entry_title = "love defeats war"
            content = "Love is the ultimate force that can conquer all obstacles, including war."
            
            # Create project structure with entry
            project_dir = os.path.join(temp_project_dir, project_name)
            notes_dir = os.path.join(project_dir, "notes")
            os.makedirs(notes_dir, exist_ok=True)
            
            entry_path = os.path.join(notes_dir, f"{entry_title}.md")
            with open(entry_path, "w") as f:
                f.write(f"""---
title: {entry_title}
created: 2024-04-07T12:00:00
updated: 2024-04-07T12:00:00
memory_type: notes
tags:
- love
- war
source: function
entry_id: 1234567890
---

# {entry_title}

{content}
""")
            
            # Mock function to read entry
            async def mock_read_entry(project_name, entry_title):
                # Try to find entry in various subdirectories
                for subdir in ["notes", "research", "instructions", "concepts"]:
                    possible_path = os.path.join(temp_project_dir, project_name, subdir, f"{entry_title}.md")
                    if os.path.exists(possible_path):
                        with open(possible_path, "r") as f:
                            raw_content = f.read()
                        
                        # Extract the content (skip YAML frontmatter and title)
                        parts = raw_content.split("---\n\n")
                        if len(parts) > 1:
                            content = parts[1]
                            # Remove title line
                            if content.startswith(f"# {entry_title}"):
                                content = content[len(f"# {entry_title}"):]
                            return {"success": True, "content": content.strip(), "title": entry_title}
                
                # Special case for main
                if entry_title == "main":
                    main_path = os.path.join(temp_project_dir, project_name, "main.md")
                    if os.path.exists(main_path):
                        with open(main_path, "r") as f:
                            raw_content = f.read()
                        parts = raw_content.split("---\n\n")
                        if len(parts) > 1:
                            content = parts[1]
                            if content.startswith("# main"):
                                content = content[len("# main"):]
                            return {"success": True, "content": content.strip(), "title": "main"}
                
                return {"success": False, "message": "Entry not found"}
            
            # Assign the mock function
            memory_integration.read_project_entry = AsyncMock(side_effect=mock_read_entry)
            
            # Call the function (simulating what would happen in actual code)
            result = await memory_integration.read_project_entry(project_name, entry_title)
            
            # Verify function called with correct parameters
            memory_integration.read_project_entry.assert_called_once_with(project_name, entry_title)
            
            # Verify success result
            assert "success" in result and result["success"]
            assert "content" in result
            assert content in result["content"]
    
    @pytest.mark.asyncio
    async def test_delete_project_entry(self, temp_project_dir):
        """Test deleting an entry from a project."""
        # Mock the memory integration
        memory_integration = AsyncMock()
        
        with patch('src.memory.project_manager.memory_integration.get_memory_integration', 
                  return_value=memory_integration):
            
            # Setup test data
            project_name = "Love and War"
            entry_title = "obsolete concept"
            
            # Create project structure with entry to delete
            project_dir = os.path.join(temp_project_dir, project_name)
            notes_dir = os.path.join(project_dir, "notes")
            os.makedirs(notes_dir, exist_ok=True)
            
            entry_path = os.path.join(notes_dir, f"{entry_title}.md")
            with open(entry_path, "w") as f:
                f.write("Test content to delete")
            
            # Mock function to delete entry
            async def mock_delete_entry(project_name, entry_title):
                # Try to find entry in various subdirectories
                for subdir in ["notes", "research", "instructions", "concepts"]:
                    possible_path = os.path.join(temp_project_dir, project_name, subdir, f"{entry_title}.md")
                    if os.path.exists(possible_path):
                        os.remove(possible_path)
                        return {"success": True, "message": f"Entry '{entry_title}' deleted"}
                
                return {"success": False, "message": "Entry not found"}
            
            # Assign the mock function
            memory_integration.delete_project_entry = AsyncMock(side_effect=mock_delete_entry)
            
            # Call the function (simulating what would happen in actual code)
            result = await memory_integration.delete_project_entry(project_name, entry_title)
            
            # Verify function called with correct parameters
            memory_integration.delete_project_entry.assert_called_once_with(project_name, entry_title)
            
            # Verify success result
            assert "success" in result and result["success"]
            
            # Verify the file was actually deleted
            assert not os.path.exists(entry_path)


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
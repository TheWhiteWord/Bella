"""
storage.py
Async Markdown/YAML file I/O for Bella's memory system.
"""

from typing import Dict, Any

import aiofiles
import yaml
import os
from datetime import datetime

class MemoryStorage:

    async def find_file_by_memory_id(self, memory_id: str) -> str:
        """
        Search all memory folders for a file whose metadata contains the given memory_id.
        Returns the file path if found, else None.
        """
        import glob
        import os
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../memories/'))
        subfolders = ["self", "user", "others"]
        for sub in subfolders:
            folder = os.path.join(base_dir, sub)
            if not os.path.exists(folder):
                continue
            for file_path in glob.glob(os.path.join(folder, "*.md")):
                try:
                    mem = await self.load_memory(file_path)
                    meta = mem.get("metadata", {})
                    if meta.get("memory_id") == memory_id:
                        return file_path
                except Exception:
                    continue
        return None
    async def save_memory(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        Save memory content and metadata to a Markdown file with YAML frontmatter.
        Save in subfolders based on memory_type: self/, user/, or others/.
        Obsidian-compatible: tags as YAML list, memory_type as internal link.
        Returns:
            str: The file path of the saved memory.
        """
        # Use timestamp and topic for filename
        timestamp = metadata.get("timestamp") or datetime.utcnow().isoformat()
        topics = metadata.get("topics", [])
        topic_str = "-".join([t.replace(" ", "-") for t in topics])[:40] or "memory"
        fname = f"{timestamp[:19].replace(':', '-')}-{topic_str}.md"
        # Determine subfolder based on memory_type
        memory_type = metadata.get("memory_type", [])
        if isinstance(memory_type, str):
            memory_type = [memory_type]
        if "self" in memory_type:
            subfolder = "self"
        elif "user" in memory_type:
            subfolder = "user"
        else:
            subfolder = "others"
        dir_path = os.path.join(os.path.dirname(__file__), f"../../../../Bella/memories/{subfolder}/")
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, fname)

        # Format tags for Obsidian (YAML list)
        yaml_metadata = dict(metadata)
        if topics:
            yaml_metadata["tags"] = topics
        # Add memory_type as an Obsidian internal link at the top of the note
        if memory_type:
            type_links = " ".join(f"[[{t}]]" for t in memory_type)
            content = f"{type_links}\n\n" + content.strip()

        # Build markdown headings for key fields
        headings = []
        def heading(label, value, level=2):
            if value is None or value == []:
                return ""
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            return f"{'#' * level} {label}\n{value}\n"

        headings.append(heading("memory_id", metadata.get("memory_id"), level=1))
        headings.append(heading("timestamp", metadata.get("timestamp")))
        headings.append(heading("participants", metadata.get("participants")))
        headings.append(heading("topics", metadata.get("topics")))
        headings.append(heading("emotional_tone", metadata.get("emotional_tone")))
        headings.append(heading("summary", metadata.get("summary")))
        headings.append(heading("source", metadata.get("source")))
        headings.append(heading("memory_type", metadata.get("memory_type")))
        headings.append(heading("importance", metadata.get("importance")))
        headings_md = "".join(headings)

        # Write only markdown headings and content (no YAML frontmatter)
        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(headings_md)
            await f.write("\n" + content + "\n")
        return file_path

    async def load_memory(self, file_path: str) -> Dict[str, Any]:
        """
        Load memory content and metadata from a Markdown file.
        Returns:
            Dict[str, Any]: The loaded memory data.
        """
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            lines = await f.readlines()
        # Parse YAML frontmatter
        if lines[0].strip() == "---":
            yaml_lines = []
            i = 1
            while i < len(lines) and lines[i].strip() != "---":
                yaml_lines.append(lines[i])
                i += 1
            metadata = yaml.safe_load("".join(yaml_lines)) if yaml_lines else {}
            content = "".join(lines[i+1:]).strip()
        else:
            metadata = {}
            content = "".join(lines).strip()
        return {"metadata": metadata, "content": content}

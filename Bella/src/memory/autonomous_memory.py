"""Autonomous memory integration for voice assistant.

Provides a seamless memory layer that works in the background during voice interactions.
Using the standardized memory format compatible with the project-based system.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio
import os  # Needed for path manipulation

import numpy as np

# Use the central memory_manager to access adapters and processors
from .main_app_integration import memory_manager
from .project_manager.memory_integration import get_memory_integration



class AutonomousMemory:
    """Autonomous memory system that works in the background during conversations."""

    def __init__(self):
        """Initialize autonomous memory system."""
        # Standardized memory integration - already uses memory_manager indirectly
        self.memory_integration = get_memory_integration()

        # Memory threshold settings
        self.memory_threshold = 0.75  # Threshold for memory relevance (used semantically)
        self.memory_check_interval = 10  # Interval between memory checks in seconds
        self.last_memory_check = datetime.now()
        self.memory_context = []  # Currently active memory context
        self.last_recalled_memory = None  # Track last recalled memory to prevent repetition
        self.recall_count = 0  # Track number of recalls in current session
        self.max_recalls_per_session = 5  # Limit number of auto-recalls per session

    async def process_conversation_turn(
        self, user_input: str, response_text: str = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Process conversation turn for memory operations.

        Handles both memory storage (post-processing) and retrieval (pre-processing)
        using semantic understanding.

        Args:
            user_input: User input text.
            response_text: Optional response text from assistant (None for pre-processing).

        Returns:
            Tuple of (modified_response, memory_context_dict).
        """
        modified_response = response_text if response_text else ""
        memory_context = {"has_memory_context": False}

        try:
            if response_text is None:
                # PRE-PROCESSING: Check if we should retrieve memories
                # Use the refactored _should_augment_with_memory
                should_augment = await self._should_augment_with_memory(user_input)

                if should_augment and self.recall_count < self.max_recalls_per_session:
                    logging.info("Attempting to augment context with memory.")
                    # Use semantic search via the memory tools (already uses adapter)
                    from .register_memory_tools import semantic_memory_search

                    search_result = await semantic_memory_search(user_input)
                    if search_result and search_result.get('success'):
                        results = search_result.get('results', [])
                        if results:
                            # Select the most relevant result based on semantic score
                            # Assuming results are already sorted by relevance by semantic_memory_search
                            top_result = results[0]
                            memory_text = top_result.get('preview', '')
                            title = top_result.get('title', '')
                            similarity_score = top_result.get('score', 0.0)  # Assuming score is returned

                            # Skip if this is the same as last recalled memory
                            if memory_text != self.last_recalled_memory:
                                # Use the semantic score directly if available, otherwise check relevance
                                if similarity_score >= self.memory_threshold:
                                    is_relevant = True
                                    confidence_level = self._map_similarity_to_confidence(similarity_score)
                                else:
                                    # Fallback check if score isn't high enough or not present
                                    is_relevant = await self._is_memory_relevant_to_query(memory_text, user_input)
                                    if is_relevant:
                                        confidence_level = await self._calculate_memory_confidence(memory_text, user_input)
                                    else:
                                        confidence_level = "low"  # Explicitly low if relevance check fails

                                if is_relevant and confidence_level != "low":
                                    self.last_recalled_memory = memory_text
                                    self.recall_count += 1

                                    # Format memory response based on confidence
                                    if confidence_level == "high":
                                        prefix = "I recall that"
                                    elif confidence_level == "medium":
                                        prefix = "I believe"
                                    else:
                                        # Avoid augmenting if confidence is low after checks
                                        prefix = None

                                    if prefix:
                                        memory_context = {
                                            "has_memory_context": True,
                                            "memory_response": f"{prefix}: {memory_text}",
                                            "memory_source": title,
                                            "confidence": confidence_level,
                                            "score": similarity_score  # Include score if available
                                        }
                                        logging.info(f"Augmenting context with memory: {title} (Confidence: {confidence_level})")
                                else:
                                    logging.info(f"Memory found ('{title}') but deemed not relevant enough (Score: {similarity_score}, Relevant: {is_relevant}).")

                            else:
                                logging.info("Skipping memory augmentation: Same as last recalled memory.")
                        else:
                            logging.info("Semantic search returned no relevant memories.")
                    else:
                        logging.warning(f"Semantic memory search failed or returned no success: {search_result}")
                elif self.recall_count >= self.max_recalls_per_session:
                    logging.info("Skipping memory augmentation: Maximum recalls per session reached.")

            else:
                # POST-PROCESSING: Check if this conversation should be stored
                should_store, store_metadata = await self._should_store_conversation(user_input, response_text)

                if should_store:
                    logging.info("Determined conversation should be stored in memory.")
                    # Generate title using semantic topics
                    title = await self._generate_title_from_content(user_input, response_text)
                    memory_type = "conversations"  # Define memory type

                    # Create content
                    content = f"User: {user_input}\n\nAssistant: {response_text}"

                    # Use tags from should_store_conversation metadata
                    tags = store_metadata.get("tags", ["conversation", "auto-saved"])

                    # Store in standardized format via memory_integration (saves the .md file)
                    memory_result = await self.memory_integration.save_standardized_memory(
                        memory_type,
                        content,
                        title,
                        tags=tags
                    )

                    if memory_result and memory_result.get('success'):
                        saved_path = memory_result.get('path')
                        logging.info(f"Successfully stored conversation memory file: {saved_path}")
                        memory_context["memory_stored"] = True
                        memory_context["memory_path"] = saved_path

                        # --- Add to ChromaDB ---
                        if saved_path:
                            # Create a unique ID for ChromaDB (e.g., 'conversations/title-slug')
                            # Ensure title is filesystem-safe for ID generation
                            safe_title = re.sub(r'[^\w\-]+', '-', title.lower())
                            memory_id = f"{memory_type}/{safe_title}"

                            # Prepare metadata for ChromaDB
                            chroma_metadata = {
                                "file_path": saved_path,
                                "title": title,
                                "tags": tags,  # Pass the list directly, _add_memory handles conversion
                                "created_at": datetime.now().isoformat(),
                                "memory_type": memory_type
                                # Add other relevant metadata from store_metadata if needed
                            }
                            if "importance_score" in store_metadata:
                                chroma_metadata["importance_score"] = store_metadata["importance_score"]

                            try:
                                # Call the adapter method to add to vector DB
                                await memory_manager.enhanced_adapter._add_memory_to_vector_db(
                                    memory_id=memory_id,
                                    content=content,
                                    metadata=chroma_metadata
                                )
                                logging.info(f"Successfully indexed memory '{memory_id}' in ChromaDB.")
                            except Exception as index_e:
                                logging.error(f"Failed to index memory '{memory_id}' in ChromaDB: {index_e}")
                                # Decide if this should be a critical failure
                        else:
                            logging.warning("Memory file saved successfully, but path was not returned. Cannot index in ChromaDB.")

                        # Optionally modify response only if deemed important by semantic check
                        if store_metadata.get("is_important", False):
                            if not response_text.endswith((".", "?", "!")):
                                response_text += "."
                            modified_response = f"{response_text} I've noted this."
                            logging.info("Acknowledged storing important memory in response.")
                        else:
                            modified_response = response_text  # Keep original response otherwise
                    else:
                        logging.error(f"Failed to store conversation memory file: {memory_result}")
                else:
                    logging.debug("Conversation turn did not meet criteria for memory storage.")

            return modified_response, memory_context

        except Exception as e:
            logging.exception(f"Error in process_conversation_turn: {e}")  # Use logging.exception for stack trace
            # Return original response on error, ensure memory_context indicates failure
            return response_text if response_text else "", {"error": str(e), "has_memory_context": False}

    async def _should_store_conversation(self, user_input: str, assistant_response: str) -> Tuple[bool, Dict[str, Any]]:
        """Determine if this conversation turn should be stored using semantic importance."""
        combined_text = f"User: {user_input}\nAssistant: {assistant_response}"
        metadata = {"tags": ["conversation", "auto-saved"], "is_important": False}

        # Basic filter: Don't store very short exchanges - but check user input separately
        # To allow storing important user messages even if the response is short
        if len(user_input.split()) < 4:
            logging.debug("Skipping memory storage: User input too short.")
            return False, {}

        try:
            # Use EnhancedMemoryAdapter's should_store_memory which uses semantic scoring
            # Assuming memory_manager provides access to the adapter instance
            should_store, importance_score = await memory_manager.enhanced_adapter.should_store_memory(combined_text)

            if should_store:
                logging.info(f"Semantic check indicates conversation should be stored (Score: {importance_score:.2f}).")
                metadata["importance_score"] = importance_score
                # Add tags based on importance or detected topics
                if importance_score >= 0.85:  # Example threshold for 'important' tag
                    metadata["tags"].append("important")
                    metadata["is_important"] = True
                elif importance_score > 0.7:
                    metadata["tags"].append("detailed")

                # Extract semantic topics and add as tags
                topics = await self._extract_potential_topics(combined_text)
                for topic in topics:
                    clean_topic = topic.lower().replace(" ", "-")
                    if clean_topic and clean_topic not in metadata["tags"]:
                        metadata["tags"].append(clean_topic)

                return True, metadata
            else:
                logging.debug(f"Semantic check indicates conversation not important enough to store (Score: {importance_score:.2f}).")
                return False, {}

        except Exception as e:
            logging.error(f"Error during semantic importance check in _should_store_conversation: {e}")
            # Fallback: Store if explicitly told to remember or if it's long
            if "remember this" in combined_text.lower() or "make a note" in combined_text.lower():
                metadata["tags"].append("important")
                metadata["is_important"] = True
                return True, metadata
            if len(combined_text.split()) > 100:
                metadata["tags"].append("detailed")
                return True, metadata
            return False, {}  # Default to false on error

    async def _generate_title_from_content(self, user_input: str, assistant_response: str) -> str:
        """Generate a title using semantic topic extraction."""
        combined_text = f"User: {user_input}\nAssistant: {assistant_response}"
        try:
            # Use semantic topic detection
            topics = await self._extract_potential_topics(combined_text)

            if topics:
                # Create title from top 1-2 topics
                title = f"Discussion about {topics[0]}"
                if len(topics) > 1:
                    title += f" and {topics[1]}"
                logging.debug(f"Generated title from semantic topics: '{title}'")
                # Ensure title is reasonably short for use in IDs/filenames
                if len(title) > 60:
                    title = title[:57] + "..."
                return title

            # Fallback: Check for questions (less reliant on specific keywords)
            user_input_stripped = user_input.strip()
            if user_input_stripped.endswith("?") and len(user_input_stripped.split()) > 3:
                # Use the question itself, truncated
                words = user_input_stripped.split()
                limit = min(10, len(words))
                title = ' '.join(words[:limit]).replace("?", "") + "?"
                logging.debug(f"Generated title from question: '{title}'")
                return title

            # Fallback: Use beginning of user input
            user_words = user_input_stripped.split()
            if len(user_words) >= 3:
                limit = min(7, len(user_words))
                title = f"{' '.join(user_words[:limit])}..."
                logging.debug(f"Generated title from user input start: '{title}'")
                return title

        except Exception as e:
            logging.error(f"Error generating title semantically: {e}")

        # Last resort: Timestamp-based title
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        title = f"Conversation on {timestamp}"
        logging.debug(f"Generated timestamp title as fallback: '{title}'")
        return title

    async def _extract_potential_topics(self, text: str) -> List[str]:
        """Extract potential topics using semantic understanding."""
        try:
            # Use EnhancedMemoryAdapter's detect_memory_topics
            topics = await memory_manager.enhanced_adapter.detect_memory_topics(text, top_n=3)
            logging.debug(f"Extracted semantic topics: {topics}")
            # Basic filtering (optional, adapter might handle this)
            topics = [t for t in topics if len(t) > 2 and t.lower() not in {"user", "assistant"}]
            return topics
        except Exception as e:
            logging.error(f"Error extracting semantic topics: {e}")
            return []  # Return empty list on error

    async def _is_memory_relevant_to_query(self, memory: str, query: str) -> bool:
        """Check memory relevance using semantic similarity."""
        try:
            # Use EnhancedMemoryAdapter's compare_memory_similarity
            similarity_score = await memory_manager.enhanced_adapter.compare_memory_similarity(memory, query)
            logging.debug(f"Semantic similarity score between query and memory: {similarity_score:.4f}")
            return similarity_score >= self.memory_threshold
        except Exception as e:
            logging.error(f"Error calculating semantic similarity for relevance: {e}")
            # Fallback to basic keyword check (less reliable)
            query_words = set(query.lower().split())
            memory_words = set(memory.lower().split())
            return len(query_words.intersection(memory_words)) > 2  # Simple overlap check

    async def _should_augment_with_memory(self, query: str) -> bool:
        """Determine if context should be augmented based on query intent and timing."""
        now = datetime.now()
        # Check time interval first
        if (now - self.last_memory_check).total_seconds() < self.memory_check_interval:
            logging.debug("Skipping memory check: Within check interval.")
            return False

        self.last_memory_check = now
        query_lower = query.lower()

        # --- Simplified Intent Checks ---
        # 1. Explicit memory recall phrases (high confidence)
        explicit_recall_indicators = [
            "remember what i told you ", "recall what i said ", "what did i tell you about ",
            "what did i say about ", "what have i told you ", "do you remember when i ",
            "remember when i mentioned ", "recall when i told you ", "what was my opinion on ",
            "what did i think about ", "remind me about "
        ]
        logging.debug(f"--- Checking Explicit Indicators --- Query: '{query_lower}'")
        matched_explicit = False
        for indicator in explicit_recall_indicators:
            is_match = query_lower.startswith(indicator)
            logging.debug(f"Checking if '{query_lower}' starts with '{indicator}' -> {is_match}")
            if is_match:
                logging.debug(f"MATCH FOUND: Query starts with explicit memory recall indicator: '{indicator}' in '{query_lower}'")
                matched_explicit = True
                break

        logging.debug(f"Explicit match result: {matched_explicit}")
        if matched_explicit:
            logging.debug("Returning True due to explicit match.")
            return True
        logging.debug("--- Finished Explicit Indicators Check ---")

        # 2. Queries about personal preferences, thoughts, or past statements (medium confidence)
        # More lenient regex patterns without word boundaries
        personal_query_patterns = [
            r"my\s+(opinion|thought|view|preference|stance|position|take|feeling|belief|perspective|interpretation|understanding)",
            r"(what|how)\s+(do|did|would)\s+i\s+(think|believe|feel|say|interpret)",
            r"(what|how|what's)\s+(is|was|were|'s)\s+my\s+(opinion|thought|view|preference|stance|position|take|feeling|belief|perspective|interpretation|understanding)",
            r"(do|did)\s+i\s+(like|enjoy|prefer|agree|mention|say)",
            r"regarding\s+our\s+last\s+discussion",
            r"based\s+on\s+what\s+i\s+said",
            r"what\s+was\s+my" # Add specific case that was failing
        ]
        
        # Log before pattern matching for debugging
        logging.debug(f"Checking personal query patterns against: '{query_lower}'")
        
        for pattern in personal_query_patterns:
            if re.search(pattern, query_lower):
                logging.debug(f"Query matched pattern '{pattern}': '{query_lower}'")
                return True
                
        logging.debug(f"Query did not match any personal query patterns")
        

        # 3. General questions that *might* benefit from memory (lower confidence - rely on semantic search relevance)
        question_words = {"what", "how", "why", "when", "where", "who", "did", "do", "was", "is"}
        context_terms = {"about", "regarding", "concerning", "on", "related to", "discussion"}
        query_words = set(query_lower.split())
        has_question = bool(question_words.intersection(query_words))
        has_context_term = bool(context_terms.intersection(query_words))
        if has_question and has_context_term:
             logging.debug("Query is a question with context terms, proceeding to semantic search check.")
             return True

        # --- Avoid Augmenting General Knowledge Queries ---
        general_knowledge_patterns = [
            r"^(what|who)\s+(is|are|was|were)\s+(a|an|the)?\s*([a-z\s]+)\??$",
            r"^(tell|explain|describe)\s+(me\s+)?about\s+(a|an|the)?\s*([a-z\s]+)\??$",
            r"^define\s+([a-z\s]+)\??$"
        ]
        is_general_query = any(re.match(pattern, query_lower) for pattern in general_knowledge_patterns)
        lacks_personal_terms = not any(term in query_lower for term in ["my", "i said", "i told", "i think", "remember", "recall"])
        if is_general_query and lacks_personal_terms:
            logging.debug("Query identified as general knowledge request, skipping memory augmentation.")
            return False

        # Default: Do not augment unless specific indicators are met.
        logging.debug("Query did not meet criteria for memory augmentation.")
        return False

    async def _calculate_memory_confidence(self, memory: str, query: str) -> str:
        """Calculate confidence level using semantic similarity."""
        try:
            # Use EnhancedMemoryAdapter's compare_memory_similarity
            similarity = await memory_manager.enhanced_adapter.compare_memory_similarity(memory, query)
            return self._map_similarity_to_confidence(similarity)
        except Exception as e:
            logging.error(f"Error calculating semantic similarity for confidence: {e}")
            # Fallback (optional): could use a simpler method or default to low
            return "low"  # Default to low confidence on error

    def _map_similarity_to_confidence(self, similarity: float) -> str:
        """Maps a raw similarity score (0-1) to confidence levels."""
        if similarity > 0.8:
            return "high"
        elif similarity >= self.memory_threshold:  # Use the instance threshold
            return "medium"
        else:
            return "low"

    def reset_session_recalls(self):
        """Resets the recall counter for a new session."""
        logging.info("Resetting memory recall count for new session.")
        self.recall_count = 0
        self.last_recalled_memory = None  # Also clear the last recalled memory
"""Conversation storage service using PostgreSQL."""

import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import structlog
from sqlalchemy import create_engine, desc, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from models.conversation import Base, Conversation, ConversationMessage, ConversationSummary
from config import BackendConfig

logger = structlog.get_logger()


class ConversationService:
    """Service for managing conversations in PostgreSQL."""

    def __init__(self, config: BackendConfig):
        self.config = config
        self.engine = None
        self.SessionLocal = None
        self.is_initialized = False

    async def initialize(self):
        """Initialize the conversation service."""
        try:
            # Create database engine
            self.engine = create_engine(
                self.config.database_url,
                pool_pre_ping=True,
                pool_recycle=300,
                echo=False  # Set to True for SQL debugging
            )

            # Create tables
            Base.metadata.create_all(bind=self.engine)

            # Create session factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

            self.is_initialized = True
            logger.info("Conversation service initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize conversation service", error=str(e))
            raise

    def get_session(self) -> Session:
        """Get a database session."""
        if not self.is_initialized:
            raise RuntimeError("Conversation service not initialized")
        return self.SessionLocal()

    async def create_conversation(self, conversation_id: Optional[str] = None,
                                user_id: Optional[str] = None,
                                title: Optional[str] = None) -> str:
        """Create a new conversation."""
        try:
            if not conversation_id:
                conversation_id = str(uuid.uuid4())

            with self.get_session() as session:
                # Check if conversation already exists
                existing = session.query(Conversation).filter(
                    Conversation.id == conversation_id
                ).first()

                if existing:
                    return conversation_id

                # Create new conversation
                conversation = Conversation(
                    id=conversation_id,
                    user_id=user_id,
                    title=title or "New Conversation",
                    metadata=json.dumps({"auto_generated": True})
                )

                session.add(conversation)
                session.commit()

                logger.info("Created new conversation", conversation_id=conversation_id)
                return conversation_id

        except Exception as e:
            logger.error("Failed to create conversation", error=str(e))
            raise

    async def add_message(self, conversation_id: str, role: str, content: str,
                         sources: Optional[List[Dict]] = None,
                         processing_mode: Optional[str] = None,
                         model_used: Optional[str] = None,
                         response_time_ms: Optional[int] = None) -> int:
        """Add a message to a conversation."""
        try:
            with self.get_session() as session:
                # Ensure conversation exists
                conversation = session.query(Conversation).filter(
                    Conversation.id == conversation_id
                ).first()

                if not conversation:
                    # Auto-create conversation
                    await self.create_conversation(conversation_id)
                    # Re-fetch the conversation in this session
                    conversation = session.query(Conversation).filter(
                        Conversation.id == conversation_id
                    ).first()

                # Create message
                message = ConversationMessage(
                    conversation_id=conversation_id,
                    role=role,
                    content=content,
                    sources=json.dumps(sources) if sources else None,
                    processing_mode=processing_mode,
                    model_used=model_used,
                    token_count=len(content.split()) if content else 0,  # Rough estimate
                    response_time_ms=response_time_ms
                )

                session.add(message)

                # Update conversation timestamp if conversation exists
                if conversation:
                    conversation.updated_at = func.now()

                # Auto-generate title from first user message
                if role == "user" and (not conversation.title or conversation.title == "New Conversation"):
                    title = content[:100] + "..." if len(content) > 100 else content
                    conversation.title = title

                session.commit()

                logger.info("Added message to conversation",
                           conversation_id=conversation_id,
                           role=role,
                           message_id=message.id)

                return message.id

        except Exception as e:
            logger.error("Failed to add message", error=str(e))
            raise

    async def get_conversation_context(self, conversation_id: str,
                                     max_messages: int = 10,
                                     include_sources: bool = False) -> str:
        """Get conversation context for AI processing."""
        try:
            with self.get_session() as session:
                # Get recent messages
                messages = session.query(ConversationMessage).filter(
                    ConversationMessage.conversation_id == conversation_id
                ).order_by(desc(ConversationMessage.created_at)).limit(max_messages).all()

                if not messages:
                    return ""

                # Reverse to get chronological order
                messages = list(reversed(messages))

                context_parts = []
                for message in messages:
                    # Add basic message
                    if message.role == "user":
                        context_parts.append(f"User: {message.content}")
                    else:
                        context_parts.append(f"Assistant: {message.content}")

                        # Optionally include sources for assistant messages
                        if include_sources and message.sources:
                            try:
                                sources = json.loads(message.sources)
                                if sources:
                                    source_info = f"[Sources: {len(sources)} documents]"
                                    context_parts.append(source_info)
                            except:
                                pass

                context = "\n".join(context_parts)
                logger.info("Retrieved conversation context",
                           conversation_id=conversation_id,
                           message_count=len(messages),
                           context_length=len(context))

                return context

        except Exception as e:
            logger.error("Failed to get conversation context", error=str(e))
            return ""

    async def get_conversation_history(self, conversation_id: str,
                                     limit: int = 50) -> List[Dict[str, Any]]:
        """Get full conversation history for display."""
        try:
            with self.get_session() as session:
                messages = session.query(ConversationMessage).filter(
                    ConversationMessage.conversation_id == conversation_id
                ).order_by(ConversationMessage.created_at).limit(limit).all()

                history = []
                for message in messages:
                    message_data = {
                        "id": message.id,
                        "role": message.role,
                        "content": message.content,
                        "created_at": message.created_at.isoformat(),
                        "processing_mode": message.processing_mode,
                        "model_used": message.model_used,
                        "response_time_ms": message.response_time_ms
                    }

                    # Add sources for assistant messages
                    if message.sources:
                        try:
                            message_data["sources"] = json.loads(message.sources)
                        except:
                            message_data["sources"] = []

                    history.append(message_data)

                logger.info("Retrieved conversation history",
                           conversation_id=conversation_id,
                           message_count=len(history))

                return history

        except Exception as e:
            logger.error("Failed to get conversation history", error=str(e))
            return []

    async def get_user_conversations(self, user_id: Optional[str] = None,
                                   limit: int = 20) -> List[Dict[str, Any]]:
        """Get list of conversations for a user."""
        try:
            with self.get_session() as session:
                query = session.query(Conversation).filter(
                    Conversation.is_active == True
                )

                if user_id:
                    query = query.filter(Conversation.user_id == user_id)

                conversations = query.order_by(desc(Conversation.updated_at)).limit(limit).all()

                result = []
                for conv in conversations:
                    # Get message count
                    message_count = session.query(ConversationMessage).filter(
                        ConversationMessage.conversation_id == conv.id
                    ).count()

                    # Get last message preview
                    last_message = session.query(ConversationMessage).filter(
                        ConversationMessage.conversation_id == conv.id
                    ).order_by(desc(ConversationMessage.created_at)).first()

                    last_message_preview = ""
                    if last_message:
                        last_message_preview = last_message.content[:100] + "..." if len(last_message.content) > 100 else last_message.content

                    result.append({
                        "id": conv.id,
                        "title": conv.title,
                        "created_at": conv.created_at.isoformat(),
                        "updated_at": conv.updated_at.isoformat(),
                        "message_count": message_count,
                        "last_message_preview": last_message_preview
                    })

                return result

        except Exception as e:
            logger.error("Failed to get user conversations", error=str(e))
            return []

    async def clear_conversation(self, conversation_id: str) -> bool:
        """Clear a conversation (soft delete)."""
        try:
            with self.get_session() as session:
                conversation = session.query(Conversation).filter(
                    Conversation.id == conversation_id
                ).first()

                if conversation:
                    conversation.is_active = False
                    session.commit()
                    logger.info("Cleared conversation", conversation_id=conversation_id)
                    return True
                else:
                    logger.warning("Conversation not found for clearing", conversation_id=conversation_id)
                    return False

        except Exception as e:
            logger.error("Failed to clear conversation", error=str(e))
            return False

    async def delete_old_conversations(self, days_old: int = 30) -> int:
        """Delete conversations older than specified days."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)

            with self.get_session() as session:
                # Get conversations to delete
                old_conversations = session.query(Conversation).filter(
                    Conversation.updated_at < cutoff_date,
                    Conversation.is_active == True
                ).all()

                count = 0
                for conv in old_conversations:
                    # Delete messages
                    session.query(ConversationMessage).filter(
                        ConversationMessage.conversation_id == conv.id
                    ).delete()

                    # Delete summaries
                    session.query(ConversationSummary).filter(
                        ConversationSummary.conversation_id == conv.id
                    ).delete()

                    # Delete conversation
                    session.delete(conv)
                    count += 1

                session.commit()
                logger.info("Deleted old conversations", count=count, days_old=days_old)
                return count

        except Exception as e:
            logger.error("Failed to delete old conversations", error=str(e))
            return 0

    async def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        try:
            with self.get_session() as session:
                total_conversations = session.query(Conversation).filter(
                    Conversation.is_active == True
                ).count()

                total_messages = session.query(ConversationMessage).count()

                # Recent activity (last 7 days)
                week_ago = datetime.utcnow() - timedelta(days=7)
                recent_conversations = session.query(Conversation).filter(
                    Conversation.created_at >= week_ago,
                    Conversation.is_active == True
                ).count()

                recent_messages = session.query(ConversationMessage).filter(
                    ConversationMessage.created_at >= week_ago
                ).count()

                return {
                    "total_conversations": total_conversations,
                    "total_messages": total_messages,
                    "recent_conversations_7d": recent_conversations,
                    "recent_messages_7d": recent_messages,
                    "service_status": "healthy" if self.is_initialized else "uninitialized"
                }

        except Exception as e:
            logger.error("Failed to get conversation stats", error=str(e))
            return {"service_status": "error", "error": str(e)}

    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.engine:
                self.engine.dispose()
            self.is_initialized = False
            logger.info("Conversation service cleanup completed")
        except Exception as e:
            logger.error("Error during conversation service cleanup", error=str(e))
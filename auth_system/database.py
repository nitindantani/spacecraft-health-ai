"""
database.py - SQLite database for Spacecraft Health AI
Install: pip install sqlalchemy python-jose[cryptography]
"""
import os, hashlib, secrets
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    DateTime, Text, Boolean, ForeignKey, JSON
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# Save DB in project root (one level up from auth_system/)
DB_PATH      = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "spacecraft_health.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── Models ────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"
    id            = Column(Integer, primary_key=True, index=True)
    username      = Column(String(50),  unique=True, nullable=False, index=True)
    email         = Column(String(100), unique=True, nullable=True)
    password_hash = Column(String(256), nullable=False)
    full_name     = Column(String(100), nullable=True)
    role          = Column(String(20),  default="operator")
    is_active     = Column(Boolean,     default=True)
    created_at    = Column(DateTime,    default=lambda: datetime.now(timezone.utc))
    last_login    = Column(DateTime,    nullable=True)
    messages      = relationship("ChatMessage",  back_populates="user", cascade="all, delete")
    events        = relationship("AnomalyEvent", back_populates="user", cascade="all, delete")


class ChatMessage(Base):
    __tablename__  = "chat_messages"
    id             = Column(Integer, primary_key=True, index=True)
    user_id        = Column(Integer, ForeignKey("users.id"), nullable=False)
    role           = Column(String(10),  nullable=False)
    message        = Column(Text,        nullable=False)
    message_type   = Column(String(20),  default="text")
    anomaly_score  = Column(Float,       nullable=True)
    mission_status = Column(String(20),  nullable=True)
    timestamp      = Column(DateTime,    default=lambda: datetime.now(timezone.utc), index=True)
    user           = relationship("User", back_populates="messages")


class AnomalyEvent(Base):
    __tablename__  = "anomaly_events"
    id             = Column(Integer, primary_key=True, index=True)
    user_id        = Column(Integer, ForeignKey("users.id"), nullable=False)
    event_type     = Column(String(50),  nullable=False)
    anomaly_score  = Column(Float,       nullable=False)
    mission_status = Column(String(20),  nullable=False)
    dominant_cause = Column(String(30),  nullable=True)
    cnn_event      = Column(String(50),  nullable=True)
    stalta_ratio   = Column(Float,       nullable=True)
    sensor_data    = Column(JSON,        nullable=True)
    diagnosis      = Column(Text,        nullable=True)
    risk_level     = Column(String(20),  nullable=True)
    timestamp      = Column(DateTime,    default=lambda: datetime.now(timezone.utc), index=True)
    user           = relationship("User", back_populates="events")


# ── Helpers ───────────────────────────────────────────────────
def init_db():
    Base.metadata.create_all(bind=engine)
    print(f"[DB] Initialized: {DB_PATH}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_password(password: str) -> str:
    salt   = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{hashed}"

def verify_password(password: str, stored: str) -> bool:
    try:
        salt, hashed = stored.split(":", 1)
        return hashlib.sha256((salt + password).encode()).hexdigest() == hashed
    except Exception:
        return False

def create_user(db, username, password, email=None, full_name=None):
    if db.query(User).filter(User.username == username).first():
        return None
    if email and db.query(User).filter(User.email == email).first():
        return None
    user = User(username=username, email=email,
                password_hash=hash_password(password),
                full_name=full_name or username)
    db.add(user); db.commit(); db.refresh(user)
    return user

def get_user(db, username):
    return db.query(User).filter(User.username == username).first()

def authenticate_user(db, username, password):
    user = get_user(db, username)
    if not user or not verify_password(password, user.password_hash):
        return None
    user.last_login = datetime.now(timezone.utc)
    db.commit()
    return user

def save_message(db, user_id, role, message, msg_type="text",
                 score=None, status=None):
    msg = ChatMessage(user_id=user_id, role=role,
                      message=message[:4000], message_type=msg_type,
                      anomaly_score=score, mission_status=status)
    db.add(msg); db.commit(); db.refresh(msg)
    return msg

def get_chat_history(db, user_id, limit=200):
    return (db.query(ChatMessage)
              .filter(ChatMessage.user_id == user_id)
              .order_by(ChatMessage.timestamp.asc())
              .limit(limit).all())

def save_anomaly_event(db, user_id, event_type, score, status,
                       cause=None, cnn_event=None, stalta=None,
                       sensors=None, diagnosis=None, risk=None):
    ev = AnomalyEvent(user_id=user_id, event_type=event_type,
                      anomaly_score=score, mission_status=status,
                      dominant_cause=cause, cnn_event=cnn_event,
                      stalta_ratio=stalta, sensor_data=sensors,
                      diagnosis=diagnosis, risk_level=risk)
    db.add(ev); db.commit(); db.refresh(ev)
    return ev

def get_anomaly_history(db, user_id, limit=200):
    return (db.query(AnomalyEvent)
              .filter(AnomalyEvent.user_id == user_id)
              .order_by(AnomalyEvent.timestamp.desc())
              .limit(limit).all())

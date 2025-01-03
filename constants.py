import enum


class SecurityThreshold(enum.Enum):
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.8


class SessionStatus(enum.Enum):
    PENDING = "pending"
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"

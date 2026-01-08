"""
Validation message types.

Provides structured validation results that can be displayed to users
and serialized for frontend communication.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class Severity(Enum):
    """Message severity levels."""
    ERROR = "error"      # Fatal: cannot proceed
    WARNING = "warning"  # Non-fatal: can proceed but suboptimal
    INFO = "info"        # Informational: no action needed


@dataclass
class ValidationMessage:
    """Single validation message.

    Attributes:
        severity: Message severity level
        code: Machine-readable error code
        message: Human-readable message
        field: Parameter field that triggered the message (optional)
        suggestion: Suggested fix or improvement (optional)
        details: Additional details dictionary (optional)
    """
    severity: Severity
    code: str
    message: str
    field: Optional[str] = None
    suggestion: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'severity': self.severity.value,
            'code': self.code,
            'message': self.message,
        }
        if self.field:
            result['field'] = self.field
        if self.suggestion:
            result['suggestion'] = self.suggestion
        if self.details:
            result['details'] = self.details
        return result

    @classmethod
    def error(
        cls,
        code: str,
        message: str,
        field: Optional[str] = None,
        suggestion: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> 'ValidationMessage':
        """Create an error message."""
        return cls(
            severity=Severity.ERROR,
            code=code,
            message=message,
            field=field,
            suggestion=suggestion,
            details=details
        )

    @classmethod
    def warning(
        cls,
        code: str,
        message: str,
        field: Optional[str] = None,
        suggestion: Optional[str] = None
    ) -> 'ValidationMessage':
        """Create a warning message."""
        return cls(
            severity=Severity.WARNING,
            code=code,
            message=message,
            field=field,
            suggestion=suggestion
        )

    @classmethod
    def info(
        cls,
        code: str,
        message: str,
        field: Optional[str] = None
    ) -> 'ValidationMessage':
        """Create an info message."""
        return cls(
            severity=Severity.INFO,
            code=code,
            message=message,
            field=field
        )


@dataclass
class ValidationResult:
    """Complete validation result.

    Attributes:
        is_valid: Whether validation passed (no errors)
        messages: List of all validation messages
    """
    is_valid: bool = True
    messages: List[ValidationMessage] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationMessage]:
        """Get only error messages."""
        return [m for m in self.messages if m.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[ValidationMessage]:
        """Get only warning messages."""
        return [m for m in self.messages if m.severity == Severity.WARNING]

    @property
    def infos(self) -> List[ValidationMessage]:
        """Get only info messages."""
        return [m for m in self.messages if m.severity == Severity.INFO]

    def add_error(
        self,
        code: str,
        message: str,
        field: Optional[str] = None,
        suggestion: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an error message and mark as invalid."""
        self.messages.append(ValidationMessage.error(
            code=code,
            message=message,
            field=field,
            suggestion=suggestion,
            details=details
        ))
        self.is_valid = False

    def add_warning(
        self,
        code: str,
        message: str,
        field: Optional[str] = None,
        suggestion: Optional[str] = None
    ) -> None:
        """Add a warning message."""
        self.messages.append(ValidationMessage.warning(
            code=code,
            message=message,
            field=field,
            suggestion=suggestion
        ))

    def add_info(
        self,
        code: str,
        message: str,
        field: Optional[str] = None
    ) -> None:
        """Add an info message."""
        self.messages.append(ValidationMessage.info(
            code=code,
            message=message,
            field=field
        ))

    def merge(self, other: 'ValidationResult') -> None:
        """Merge another validation result into this one."""
        self.messages.extend(other.messages)
        if not other.is_valid:
            self.is_valid = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'is_valid': self.is_valid,
            'errors': [m.to_dict() for m in self.errors],
            'warnings': [m.to_dict() for m in self.warnings],
            'infos': [m.to_dict() for m in self.infos],
        }

    @classmethod
    def success(cls) -> 'ValidationResult':
        """Create a successful (empty) validation result."""
        return cls(is_valid=True, messages=[])

    @classmethod
    def failure(cls, error: ValidationMessage) -> 'ValidationResult':
        """Create a failed validation result with a single error."""
        return cls(is_valid=False, messages=[error])

    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.is_valid

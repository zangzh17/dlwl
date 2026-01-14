"""
Wizard layer for parameter generation.

Wizards convert user-friendly input parameters to structured parameters
suitable for optimization. Each DOE type has a specialized wizard.

Main entry points:
- create_wizard(doe_type): Get wizard instance for specific DOE type
- generate_params(user_input): Convenience function to generate params directly
"""

from .base import (
    BaseWizard,
    WizardOutput,
)
from .splitter import SplitterWizard
from .diffuser import DiffuserWizard
from .lens import LensWizard
from .custom import CustomPatternWizard
from .factory import create_wizard, generate_params

__all__ = [
    'BaseWizard',
    'WizardOutput',
    'SplitterWizard',
    'DiffuserWizard',
    'LensWizard',
    'CustomPatternWizard',
    'create_wizard',
    'generate_params',
]

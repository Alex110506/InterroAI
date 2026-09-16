"""The in-memory `Accounts` the sign-in tests run on, held to the shared contract."""
from __future__ import annotations

import pytest
from fakes.accounts import InMemoryAccounts
from port_contracts.accounts import AccountsContract


class TestInMemoryAccounts(AccountsContract):
    @pytest.fixture
    def accounts(self):
        return InMemoryAccounts()

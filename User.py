from __future__ import annotations

class User:
    def __init__(self: User, name: str) -> None:
        self.full_name = name
        self.first_name = self.full_name.split()[0]
        try:
            self.last_name = self.full_name.split()[1]
        except IndexError:
            self.last_name = ''

        self.symptoms = ''
        self.diagnosis: str
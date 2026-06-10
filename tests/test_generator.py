import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate import Generator, GeneratorConfig


class GeneratorMessageTest(unittest.TestCase):
    def test_build_message_last_message_contains_assistant_reply(self) -> None:
        generator = Generator(GeneratorConfig())
        messages = generator.build_message(
            system_prompt="SYSTEMPROMPT",
            reply="Here is the assistant reply.",
        )

        self.assertEqual(
            messages[-1],
            {
                "role": "user",
                "content": "THE ASSISTANT REPLY:\nHere is the assistant reply.",
            },
        )
        
        print(f"Generated messages: {messages}")

if __name__ == "__main__":
    unittest.main()

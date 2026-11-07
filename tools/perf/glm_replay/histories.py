"""Newly authored Apache-2.0 public workload sources, including reproducible Pi-style histories."""
from .common import canonical_bytes

TOPICS = (
    ("harbor", "A harbor team records tidal heights, checks mooring ropes, and plans a safe survey."),
    ("orchard", "An orchard team maps flowering dates, labels grafts, and compares soil moisture."),
    ("archive", "An archive team repairs paper folders, records catalog numbers, and rotates exhibits."),
    ("trail", "A trail team surveys wooden bridges, counts signposts, and schedules maintenance."),
)
TOOLS = [{"type": "function", "function": {"name": "read", "description": "Read a public example file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
         {"type": "function", "function": {"name": "bash", "description": "Describe an offline example command",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]


def source_material(family, stream_id, repetitions=8):
    topic, sentence = TOPICS[stream_id]
    if family == "prose":
        paragraphs = [f"{topic.title()} field notebook, section {i}. {sentence} "
                      "The morning record states what was observed, how the measurement was taken, "
                      "and which questions remain open. At noon the group compares independent notes. "
                      "A small discrepancy is recorded with its units before any conclusion is drawn."
                      for i in range(repetitions)]
        body = "\n\n".join(paragraphs)
        messages = [{"role": "system", "content": f"Public {topic} notebook stream {stream_id}."},
                    {"role": "user", "content": body + "\nContinue the notebook in the same concrete style."}]
        tape = (f"The {topic} team next compared the dated observations and wrote a careful explanation. "
                "Each entry retained the original units and the method used to check the result. ") * repetitions
        text = body
        tools = None
    elif family == "code":
        sections = [f"def {topic}_total_{i}(rows):\n"
                    f"    \"\"\"Sum validated {topic} counts for batch {i}.\"\"\"\n"
                    "    total = 0\n    for row in rows:\n"
                    "        count = row['count']\n"
                    "        if not isinstance(count, int) or count < 0:\n"
                    "            raise ValueError('count must be a nonnegative integer')\n"
                    "        total += count\n    return total\n"
                    for i in range(repetitions)]
        body = "\n".join(sections)
        messages = [{"role": "system", "content": f"Public Python example stream {stream_id}, {topic}."},
                    {"role": "user", "content": body + "\nAdd a compact usage example."}]
        tape = (f"\ndef test_{topic}_counts():\n"
                f"    assert {topic}_total_0([{{'count': 3}}, {{'count': 5}}]) == 8\n"
                f"    assert {topic}_total_0([]) == 0\n") * repetitions
        text = body
        tools = None
    else:
        messages = [{"role": "system", "content": f"Public Pi-style tool history stream {stream_id}: {topic}."}]
        for i in range(repetitions):
            read_id, test_id = f"read_{i}", f"test_{i}"
            messages.extend([
                {"role": "user", "content": f"Check the {topic} ledger section {i} and explain the total."},
                {"role": "assistant", "content": "I will read the example ledger.", "tool_calls": [
                    {"id": read_id, "type": "function", "function": {"name": "read",
                     "arguments": canonical_bytes({"path": f"examples/{topic}/ledger.txt"}).decode()}}]},
                {"role": "tool", "tool_call_id": read_id, "name": "read",
                 "content": f"{topic} public example {i}\ncount: 3\ncount: 5\n"},
                {"role": "assistant", "content": "I will check the two recorded counts.", "tool_calls": [
                    {"id": test_id, "type": "function", "function": {"name": "bash",
                     "arguments": '{"command":"python3 -c \'assert 3 + 5 == 8\'"}'}}]},
                {"role": "tool", "tool_call_id": test_id, "name": "bash", "content": "exit 0\n"},
                {"role": "assistant", "content": f"The {topic} ledger total is eight; the example check passed."},
            ])
        messages.append({"role": "user", "content": f"Write the next {topic} inspection note."})
        tools = TOOLS
        tape = (f"I checked the public {topic} ledger and retained the two source counts. "
                "The note records the result and the next inspection step. ") * repetitions
        text = canonical_bytes({"messages": messages, "tools": tools}).decode("ascii")
    return {"text": text, "messages": messages, "tools": tools, "tape": tape}

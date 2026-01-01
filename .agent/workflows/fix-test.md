---
description: Diagnostic path for fixing test failures in ORTHOS
---
# Fix Test Workflow

Use this workflow when `python test_orthos.py` fails.

// turbo
1. Run the tests and capture the output:
   `python test_orthos.py > test_results.log 2>&1`

2. Identify the failing test case(s) and the associated error message.

3. Examine the source code for the failing component. Use `grep` to find the implementation.

4. Check the `docs/agentic/knowledge-base.md` for "Common Traps" related to that component.

5. Apply a fix to the code.

// turbo
6. Verify the fix by running only the specific failing test (if possible) or the whole suite:
   `python test_orthos.py`

7. Update `docs/agentic/knowledge-base.md` if the fix revealed a new type of trap.

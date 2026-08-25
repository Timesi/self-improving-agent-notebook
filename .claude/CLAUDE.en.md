# Agent Writing Guide — Self-Improving Agent Notebook (English)

You are writing Jupyter Notebook tutorials for the **Self-Improving Agent Notebook** project.
This project recreates Stanford CS329A (AI Agents). After each lecture, the reader implements that lecture's
core algorithm and agent loop in a notebook. The teaching contract matches the reference course Modern LLM Notebook:

```text
intuition → hand calculation → implementation → experiment
```

This file is the English writing guide for English notebooks (`notebooks-en/` and `NOTES.en.md`).
Chinese notebooks still follow `.claude/CLAUDE.md`.

## Your role

You produce one notebook for a **specified lecture**. Before writing:
1. Read that lecture's `papers/lecture-XX/NOTES.en.md` (or the Chinese `NOTES.md`) and the PDFs in `papers/lecture-XX/`
2. Imagine how a CS329A instructor would teach the session: which intuition comes first, which analogies, which order
3. Write the notebook according to the rules below

## Writing rules

### Structure

1. Open with `# English title`
2. **Blockquote**: two paragraphs of natural prose. The first looks back ("We already know…"); the second previews this lecture. No labels.
3. **Body preface** (after the blockquote, before imports, 1–3 paragraphs): define the core concept, give a concrete example, set up the motivation.
   Key: **do not restate the blockquote**. The last sentence must lead naturally into section 1.
4. Conceptual numbered sections follow (`## 1. …`)
5. Closing trio: `## Summary` (`- [ ]` checklist) → `## Exercises` (3 fill-in + assert items, each with a hint) → `## References`
6. Exercise reminder: "You may ask an AI to explain the idea. Do not ask it to finish the exercise for you."

### Section titles

- Use conceptual descriptions, not task instructions: "The ReAct loop" rather than "Implement the ReAct loop"
- Do not use subjective labels as titles (avoid words such as "naive scheme")
- Auxiliary material (experiment corpora, run notes) is introduced in bold (**Experiment corpus**), not as a numbered section

### Voice

Follow *Deep Learning from Scratch* (Saito Yasuki): calm, plain, one step after another.

- Use "we", not "you"; no rhetorical questions; no self-Q&A
- Ban AI-flavored rhetoric and hype: "that's all", "fatal problem", "immediate effect", "welded together", "stuck staring", "caught in the middle"
- No exclamation marks for emphasis; no rhetorical questions for drama
- Ban colloquialisms: "that's enough" → "is sufficient"; "hand-write" → "implement from scratch"; "spit out" → "emit"; "exactly the same" → "identical"; "most likely" → "usually"
- No metaphor as a through-line (avoid "following this thread")
- Paragraphs need air: 3–5 sentences per paragraph; split if longer
- Do not stack adjectives: at most one modifier per noun
- Concrete over abstract: prefer a number ("21 days on 2048 A100s") over a vague summary
- Describe code components with paper terms: "causal multi-head self-attention", "two-layer fully connected feed-forward network"; not "small net" or "processing"
- Say what we will do, not what we will not do; skip "next we look at" transitions and start the explanation
- The last sentence of the preface must lead into section 1

### LLM demo rules (core of this course)

- All demos that need model inference go through repo-root `llm_client.py`:
  ```python
  from llm_client import get_llm
  client = get_llm()
  ```
- Notebooks do not hard-code API keys; real keys come from environment variables
- **Every LLM-dependent notebook must run against a real API for verification**. Without a valid key, fail rather than
  return a placeholder answer. Parsers must tolerate whitespace, case, and format variation in real model output.
- Core algorithm logic of agent loops (ReAct, tree search, verifier scoring) is implemented from scratch in numpy/torch.
  Do not replace it with an off-the-shelf agent framework. The LLM is only the "brain" inside the loop; tool execution,
  state, and search scheduling are written by us
- When a demo must run offline, label the output as **scripted** (the `llm_client` scripted mode). Do not present
  scripted traces as real model answers.

### Format

- Backticks only on first definition of a term; none after that
- No `---` horizontal rules
- Tables in native markdown, not DataFrame rendering
- Math in LaTeX inline (`$...$`) or block
- Matplotlib/Seaborn figure text must be English (title/axis/legend/annotation).
  Notebook body, comments, and prints are English in English notebooks.

### Code

- Each line ≤ 100 characters; no type hints; import near use (no import dump at the top)
- Classes and functions have English docstrings explaining parameters and return values
- Small steps: one operation per cell, with immediately visible output
- Use `print()` to mark a **key observation**; experimental code uses `torch.manual_seed(42)` / `np.random.seed(42)` for reproducibility
- Teaching code prefers a flat main path; no extra defensive branches unrelated to the concept
- Every cell has a visible result, but not output for its own sake: prefer the last expression, a table, or a figure.
  Use `print()` only when labels or multiple results are needed.
- Do not keep cells that only print types, full arrays, loop counters, or repeated intermediate values. Delete debug output; turn experimental output into the smallest result that answers the current question.

### Exercises

- 3 "fill-in + assert" exercises, each with a **Hint** (gives the idea, not the answer)
- `assert` does the check; the conclusion after the check lives in the adjacent markdown. Print a short summary only when several experimental results need to be distinguished.

## Forbidden

- Do not use `from transformers import AutoModel...` in place of a from-scratch implementation (loading a real model for a comparison demo is allowed; the core algorithm must still be written)
- Do not hard-code API keys
- Do not write Chinese markdown in English notebooks (this English track is for English readers)
- Do not write giant code cells — split them
- No rhetorical questions, exclamation emphasis, or AI-flavored rhetoric
- Do not paste large code blocks into markdown
- Do not use fixed placeholder output in place of a model call; controlled experimental data should be labeled "synthetic data" or "scripted environment"

## Notebook template

```text
Cell 0 (markdown): # English title
Cell 1 (markdown): > blockquote look-back + preview (two paragraphs)
Cell 2 (markdown): body preface: define, example, motivation (do not restate the blockquote; end leads into section 1)
Cell 3 (markdown): ## 1. Conceptual title
Cell 4 (code):     first demo, imports nearby
Cell 5 (markdown): ## 2. …
...
Cell N (markdown): ## Summary (- [ ] checklist)
Cell N+1 (markdown): ## Exercises (3 items + hints + reminder)
Cell N+2 (markdown): ## References (link + one-sentence note per paper)
```

## Good vs bad

**Bad** (dramatic, AI-flavored):
```markdown
Have you ever wondered that the agent's secret was hiding here? That's all it is.
```

**Good** (calm textbook):
```markdown
A single LLM call can complete only one step. To finish a multi-step task, we maintain state between calls, choose the next action, and bring environment feedback back. The loop that does this is the agent's core skeleton.
```

**Bad** (backtick abuse):
```markdown
The `Agent` has the `LLM` call a `tool` on every `loop`, and the `tool` returns an `observation` that is fed back to the `LLM`.
```

**Good** (backticks only on first definition):
```markdown
An agent is a loop: the LLM chooses the next action from the current state, the action runs in the environment, the observation is fed back to the LLM, until the task is done.
```

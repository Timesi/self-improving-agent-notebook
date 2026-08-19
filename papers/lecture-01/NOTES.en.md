# Lecture 01 — What is an AI Agent (study notes)

> This file is the study notes for CS329A lecture 1 (Course Overview), and the source material for `notebooks-en/part1-foundation/01-course-overview.ipynb`.
> Source: https://cs329a.stanford.edu/ (Autumn 2025 syllabus)

> **Template note**: this lecture has no assigned papers. It is an overview lecture. As required by the project, the "Close reading" section is replaced by "Concept close reading", unpacking the five core concepts this lecture must make clear.

## Lecture theme

The core question this lecture answers: **What is an AI Agent? Why is it the next form of an LLM? Along what path does this course turn an agent from a concept into something we implement by hand?**

Why it is lecture 1: it is the roadmap for the whole course. Every later lecture is a piece of this map — test-time compute (L2), verifiers (L3), tools and code feedback (L4), multi-step planning (L5), RL training (L6), open-ended evolution (L7), search and deep research (L8), post-training evolution (L9), SWE engineering (L10), memory (L11), evaluation (L14), autonomy and robotics (L15/L16). They all serve one object: making an LLM do smarter work inside a loop. We give global coordinates first, then fill in details lecture by lecture.

This lecture has three jobs:

1. Build intuition and a unified view of an agent: the LLM is the brain, the loop is the skeleton, tools and memory are the hands and feet.
2. Give a paradigm table (ReAct / Tool use / Planning / Multi-agent / Memory / Framework) so later lectures have a place on the map.
3. Give the course map (4 parts, 17 notebooks) and preview the teaching contract: intuition → hand calculation → implementation → experiment.

## Concept close reading

> The five concepts map to OUTLINE sections: concepts 1 and 2 → "From LLM applications to agents"; concept 3 → "Definition and components of an agent" and "One agent loop"; concept 4 → "Map of the agent ecosystem"; concept 5 → "Course path".

### Concept 1: What an agent is — from "one conversation" to "a goal-directed executor"

**Classical definition (Wooldridge, 1995/1999)**: an agent is an entity situated in an environment. It senses the environment through sensors and acts on it through effectors. Two keywords. First, **autonomy**: without direct external intervention, the agent chooses actions to reach a goal. Second, **situatedness**: it faces a partially observable, changing environment, not a well-formed input. This definition predates LLMs and covers software agents and robots.

**Modern LLM-based agent (survey consensus, Wang et al. 2023)**: a system whose decision core is an LLM. In a loop it calls the LLM for perception, reasoning, and decision-making, turns decisions into actions in the environment (usually via tools), and feeds action results back in until the goal is reached. The LLM here is the "brain", not the whole system.

**One-sentence definition for the reader** (reusable in the notebook):

> An agent is a loop: the LLM chooses the next action from the current state, the action runs in the environment, the observation is fed back to the LLM, until the task is done.

**Difference from a pure LLM application** — the most important watershed of this lecture:

| Axis | Pure LLM application | LLM-based Agent |
|---|---|---|
| Calling style | Single (or fixed-round) Q&A | Loop until a stop condition |
| State | None (or conversation history only) | Explicit task state and memory |
| Action | Tokens only | Executable tool calls / actions |
| Environment feedback | None | Present (execution results return to context) |
| Goal | Produce an answer | Finish a task (can fail, can retry) |

Key: an agent is not "calling an LLM in code". Auto-filling a question into a prompt and calling an API once treats the LLM as a function call; without loop, state, and action, it is still an application, not an agent.

### Concept 2: Why an agent is the next form of an LLM

A single-forward-pass LLM has four hard limits:

1. It can only "speak", not "do": it emits tokens and never touches the world; it does not know what happened in a webpage, database, file, or code executor.
2. The context window is finite: long tasks and large intermediate results do not fit in one call.
3. It does not correct itself: a single generation has no execution feedback, so it cannot know "this answer is wrong when run".
4. Knowledge is static: facts after the training cutoff cannot be obtained.

Agent-ization breaks these four limits one by one:

1. Tool calls turn "speak" into "do" — search, code execution, file I/O, GUI (L4 covers this).
2. Loops and memory break the context limit — multi-step execution, external memory (L5 planning, L11 memory).
3. Execution feedback introduces a strong "we only know after running" signal — verifiers (L3), code feedback and RLEF (L4).
4. Search / sampling / reasoning compute compensate for single-generation quality — test-time compute (L2).

In one sentence: the LLM supplies "intelligence"; the agent loop supplies "structure". Capability gains come from the loop wiring environment, tools, memory, and multi-step compute into the model, not from the model becoming stronger by itself. That is why an agent is the "next form" — it upgrades a static Q&A model into a system that can finish tasks on its own and grow with training and evolution (L6–L9 continue this line).

### Concept 3: Components of the agent loop

The minimal closed loop (in the notebook: one figure plus a loop class):

```
Observation → Decision (Reasoning + Action) → Act → Feedback → back to Observation → stop
```

1. **Observation**: assemble environment state, tool returns, and relevant memory into context. The LLM has no direct channel to the environment; everything enters as a textual observation.
2. **Decision (Reasoning + Action)**: from current state and goal, the LLM outputs "what to think + what to do next". The format can be ReAct Thought/Action or a structured tool call. This is the only part generated by the model.
3. **Action / tool execution**: parse the action and run it in the environment. Tools are the LLM's interface to the world (search, python REPL, filesystem, API, GUI).
4. **Feedback**: execution results (including errors) return to context as the next observation.
5. **Stop**: the model emits a Final Answer, or a step / budget limit is reached.

Around the loop, several things are often taught as independent modules but are really "attachments" of the loop:

- **Memory**: short-term is the message history in context; long-term is external storage (vector store / files), freeing the loop from the context window (L11).
- **Planning**: decompose the goal into subgoals, or search over action sequences (L5).
- **Verification**: add a check outside the loop so an "output" is accepted only after verification (L3).

A common confusion: an agent loop is not multi-turn chat. ChatGPT's multi-turn only "remembers turns"; there is no action and no feedback. Each round of an agent loop must pass through the environment (even if that is only reading a file). The test is whether there is an executable action that changes the world.

### Concept 4: Mainstream paradigms (agent ecosystem map)

| Paradigm | One-sentence description | Representative work / frameworks | Place in the course |
|---|---|---|---|
| ReAct | Alternate reasoning (Thought) and acting (Action); weave "think" and "do" into one trajectory | Yao et al. 2022 | L4 tools and code feedback (default skeleton of this course) |
| Tool use / Function calling | The LLM learns to emit structured tool calls; tool results become input | Toolformer (2023), OpenAI function calling, MCP | L4; MCP is a tool-interface standard |
| Planning / Search | Decompose the task first, or search the action space, rather than take one step at a time | LATS (2023), ADaPT, SPRINT | L5 multi-step planning |
| Multi-agent | Several agents collaborate, debate, divide labor; the agent becomes a society | AutoGen (2023), CrewAI, CAMEL | Cooperative forms in L6/L7 training and evolution |
| Memory-based | Explicit long-term memory lets the agent work across sessions and long tasks | MemGPT (2023), Cartridges | L11 memory |
| Agentic frameworks | Wrap loop, tools, and memory into a library for "out of the box" engineering | LangChain/LangGraph, AutoGen, Claude Agent SDK | Course-wide contrast (this course implements from scratch and does not depend on them) |
| Special forms: code-as-action / Computer-use / deep research | The agent finishes tasks by writing code, operating a GUI, or retrieving from many sources | AlphaCode, OpenAI Computer Use, deep-research agents | L8, L10, L13, L16 |

Two emphases:

1. Paradigms are not mutually exclusive. ReAct is only a decision format; it can stack planning (plan first, then ReAct), memory, and a verifier. A real agent is usually a combination.
2. The ecosystem has many "frameworks", but this course's stance is to implement the core loop from scratch (numpy/torch plus our own tool executor). Frameworks are only a contrast demo. That way we see what each component does, rather than being blocked by framework abstractions.

### Concept 5: Course path (give students the map)

Per OUTLINE.md, the course has 4 parts and 17 notebooks (Stanford L10–12 are midterm showcases with no teaching content):

- **Part 1 Foundation (L1–L5)**: start from "an LLM can only generate" and add ability layer by layer — test-time compute scaling (L2), answer verification (L3), tools and code feedback (L4), multi-step planning (L5).
- **Part 2 Training & Evolution (L6–L9)**: an agent is not hardcoded; it is trained and evolved — RL train-time scaling (L6), open-ended evolution where an agent designs agents (L7), search and deep-research agents (L8), post-training from chatbot to agent (L9).
- **Part 3 Agent Engineering (L10/L11/L14)**: turn an agent into engineering — SWE Agent (L10), memory systems (L11), evaluation and long-horizon tasks (L14).
- **Part 4 Frontiers (L12/L13/L15–L17)**: where the boundary is — LLM reasoning (L12), math agents (L13), autonomous agents (L15), multimodal robotics (L16), future directions (L17).

Teaching contract (the most important sentence of the course): from this lecture on, every concept proceeds by intuition → hand calculation → implementation → experiment. The first lecture's job is that students write the first minimal loop before class ends.

## Teaching thread (how a Stanford instructor might teach this)

The first class does not lecture papers. It does three things: build intuition, give definition and taxonomy, give the course map.

1. **Start from what students know: a single ChatGPT turn**. Live-demo a task ChatGPT cannot finish, such as "fix every stale TODO in this repo and run the tests". Point to a counter-intuitive fact: an LLM can write good code snippets but cannot finish a job by itself. The reason is that it can only speak, not do; it has no state, no tools, no loop. That sets up "LLM application vs agent".

2. **Give the agent definition**. First Wooldridge's classical definition (situated, autonomous, sense–act), then the LLM era: the LLM is the brain, the loop is the skeleton. Summarize in a reusable sentence: an agent is a loop; the LLM chooses the next action from the current state; the action runs in the environment; the observation is fed back to the LLM until the task is done.

3. **Walk one concrete task through the agent loop**. Use something like "look up the latest weather and decide whether to take an umbrella". Walk Observation → Decision → Act → Feedback by hand, and stress that every round must pass through the environment. Clarify a frequent misunderstanding: multi-turn chat is not an agent; there must be an executable action that changes the world.

4. **Give the paradigm table and ecosystem map**. List ReAct / Tool use / Planning / Multi-agent / Memory / Frameworks, each with a representative (ReAct→Yao et al., Toolformer, LATS, AutoGen, MemGPT). Stress that paradigms are not exclusive and a real agent is a combination.

5. **Give the course map**. Spread out 4 parts and 17 notebooks, and say which direction each lecture adds ability: Part 1 adds ability (reasoning compute, verification, tools, planning), Part 2 adds "how to get stronger" (training, evolution, search, post-training), Part 3 adds engineering (SWE, memory, evaluation), Part 4 adds the boundary (reasoning models, math, autonomy, robotics). Close by previewing: before class ends, we write the first minimal agent loop by hand.

Three places readers tend to get stuck:

- **Agent vs "calling an LLM in code"**: auto-filling a prompt and calling an API is not an agent; it lacks loop, state, and action. We need a contrast of "the same task, the application cannot do it, the agent can".
- **Loop vs multi-turn chat**: the test is "whether there is an action in the middle that changes the world".
- **The illusion that "an agent framework is required"**: LangChain/AutoGen are wrappers, not the thing itself. The core loop is a few dozen lines from scratch. The instructor writes the minimal loop live to break this illusion.

## Code demo ideas (3–6)

All go through `llm_client.get_llm()` (scripted mode, deterministic output, runnable without a key). Core agent-loop logic is implemented from scratch, not via an off-the-shelf framework.

1. **Minimal agent-loop skeleton (from scratch)**: write an `AgentLoop` class whose fields are message history (state), a tool registry, and an LLM. Loop body: call the LLM → parse output → if `Action`, execute the tool and append the observation to history → if `Final Answer`, stop. Use a deterministic toy environment (e.g. an add tool plus a reverse-string tool) so scripted mode can advance stably. **Key observation**: the loop structure itself is a few dozen lines; the gap between agent and application is structure, not the model.

2. **"Single call vs loop" contrast**: the same multi-step task, run with (a) a pure LLM single call (scripted mode gives a placeholder reply) and (b) an agent loop (can call tools and get environment results). Compare outputs and context length. **Key observation**: the extra ability of the agent comes from the loop and tool feedback, not from the model itself; this reproduces the watershed "the LLM application cannot, the agent can".

3. **Minimal tool-calling with llm_client**: implement `parse_action(text)` for ReAct format from scripted/real LLM output (`Thought: ...` / `Action: name(args)` / `Final Answer: ...`), plus an `execute(actions, registry)` executor, run 2–3 rounds. Action parsing must be tolerant: both scripted traces and real-model JSON must parse. **Key observation**: a tool call is essentially a text protocol; format conventions (JSON, MCP today) only standardize that protocol.

4. **Agent-component visualization (course-map figure)**: matplotlib plot of the Observation–Decision–Act–Feedback loop (English labels), hang memory, planning, and verification as "attachments" on the matching stages, and annotate each component with its lecture (tools→L4, planning→L5, verification→L3, memory→L11, training→L6–L9). **Key observation**: one figure pins all 17 lectures onto the agent loop; students see the map of the whole course.

5. **Agent-paradigm cards**: define each paradigm (ReAct / Tool use / Planning / Multi-agent / Memory / Framework) in a dict of keywords, representative work, and one applicable scene; then give a few real scenes and let students pick a paradigm. Under scripted mode this is a table plus a mapping exercise. **Key observation**: paradigms are not exclusive; a real agent is often ReAct + Planning + Memory at once.

6. **A multi-step arithmetic agent (minimal closed loop with state)**: the task is to compute `(3+5)×(7-2)`. The agent cannot finish in one shot (the scripted arithmetic tool computes one step at a time). It must call `calc("3+5")` to get 8, write it back to context, then `calc("8×5")` to get 40, then emit Final Answer. We write the tool executor; intermediate state is filled back into message history by us. **Key observation**: intermediate state is maintained by the agent itself (context write-back). That is the watershed between an agent and a single call, and the base of every later lecture.

> Implementation note: all of the above can run fully offline with deterministic tools plus a scripted LLM (per CLAUDE.md). Demo 4 is a pure matplotlib figure, demo 5 is pure data/tables, 1/2/3/6 need a parser compatible with scripted Thought/Action/Final Answer format (already built into `llm_client.py`).

## Exercise ideas (3)

1. **Complete the minimal loop**: fill in the stop condition of `AgentLoop.run(max_steps)` and the two segments "execute action → write back observation". Given a fixed task (e.g. "use a tool to compute 7+8, then reverse the result string"), `assert` that the loop emits a Final Answer within max_steps and that context length grows with rounds. Hint: there are two stop conditions, the model emits Final Answer, or steps run out; the latter should produce a readable failure message.

2. **Action parser**: fill in `parse_action(text)` supporting `Action: name(args)` / `Final Answer: ...` / no action, returning `(kind, name, args)`. Test once on a scripted-mode scripted trace and once on a real-format snippet; `assert` that edge cases (extra whitespace, multiple lines, uppercase Action) parse. Hint: look for `Final Answer` before `Action`, because the Final line should not be treated as an Action.

3. **Single call vs loop (quantify the gap)**: run the same task as a single call and as a loop; `assert` that the loop's call count is greater than 1 and that context contains tool returns, while the single call has neither; then write `count_tool_calls(trace)` to count tool calls in one round. Hint: tool returns are written back into context by us, so "the loop got tool results" is assertable and does not depend on a specific model.

## References

- [Intelligent Agents: Theory and Practice](https://www.csc.liv.ac.uk/~mjw/pubs/ker95.pdf) (Wooldridge & Jennings, The Knowledge Engineering Review, 1995) — source of the classical agent definition: situated, autonomous, sense–act loop.
- [Intelligent Agents (Multiagent Systems, chapter 1)](https://www.cs.ox.ac.uk/people/michael.wooldridge/pubs/maia-chapter.pdf) (Wooldridge, 1999) — a fuller agent definition, including BDI and rational agents.
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (Yao et al., ICLR 2023) — the reason-and-act paradigm; default agent-loop skeleton of this course.
- [A Survey on LLM-based Autonomous Agents](https://arxiv.org/abs/2308.11432) (Wang et al., 2023) — consensus summary of agent components (planning / memory / tool use).
- [The Rise and Potential of Large Language Model Based Agents: A Survey](https://arxiv.org/abs/2309.07864) (Xi et al., 2023) — a broader LLM-based agent survey and ecosystem taxonomy.
- [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) (Lilian Weng, 2023) — the clearest blog on the agent loop (planning / memory / tool use); good first extra reading.
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) (Schick et al., 2023) — early work on LLMs learning to call tools.
- [Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models](https://arxiv.org/abs/2310.04406) (Zhou et al., 2023) — planning/search paradigm (LATS).
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) (Packer et al., 2023) — memory paradigm.
- [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155) (Wu et al., 2023) — multi-agent collaboration framework.
- [Model Context Protocol](https://modelcontextprotocol.io) (Anthropic, 2024) — standard interface for tools/resources/context; the "USB-C" of the agent ecosystem.
- [CS329A Course Homepage](https://cs329a.stanford.edu/) (Stanford, Autumn 2025) — syllabus and assignment source for this course.

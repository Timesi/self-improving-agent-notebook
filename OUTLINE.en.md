# Self-Improving Agent Notebook — course outline

Recreation of Stanford CS329A (AI Agents, Autumn 2025). Organized by course lecture number; midterm showcases (L10–L12 in the original Stanford numbering) are skipped.

Part count: 4
Notebook count: 17

---

## Part 1 · Foundation — L1–L5

### 01-course-overview.ipynb — What is an AI Agent

- 1. From LLM applications to agents
- 2. Definition and components of an agent
- 3. One agent loop
- 4. Map of the agent ecosystem
- 5. Course path
- Summary
- Exercises

### 02-test-time-compute.ipynb — Test-time compute: compute we can still spend after training

- 1. Compute after training: test-time compute
- 2. Repeated sampling: Large Language Monkeys
- 3. From sampling to voting: self-consistency and best-of-n
- 4. Compute-optimal scaling: Snell's law
- 5. Combining methods and architecture search: Archon
- Summary
- Exercises
- References

### 03-robust-verification.ipynb — Answer verification: letting a model check a model

- 1. The generation–verification gap
- 2. Training a verifier: Cobbe's math verifier
- 3. Outcome reward models (ORM) and process reward models (PRM)
- 4. Step-by-step verification: Lightman
- 5. Step-level verification without human labels: Math-Shepherd
- 6. Combining verifiers with sampling
- Summary
- Exercises
- References

### 04-tool-code-feedback.ipynb — Tool use and code feedback

- 1. From reasoning to action: the ReAct loop
- 2. Defining and calling tools
- 3. Code execution as a feedback signal
- 4. RLEF: reinforcement learning from execution feedback
- 5. Constitutional AI: aligning AI with AI feedback
- Summary
- Exercises
- References

### 05-multi-step-planning.ipynb — Multi-step reasoning and planning

- 1. Limits of single-step reasoning
- 2. Task decomposition: ADaPT
- 3. Tree search: LATS
- 4. Parallel planning and execution: SPRINT
- 5. Adaptive branching: Wider or Deeper
- Summary
- Exercises
- References

---

## Part 2 · Training & Evolution — L6–L9

### 06-train-time-scaling-rl.ipynb — Train-time scaling and reinforcement learning

- 1. Train-time vs test-time scaling
- 2. STaR: bootstrapping reasoning with reasoning
- 3. DeepSeekMath and GRPO
- 4. DAPO: an open-source LLM RL system
- 5. From RL to agent training
- Summary
- Exercises
- References

### 07-open-ended-evolution.ipynb — Open-ended evolution of self-improving agents

- 1. Letting an agent design agents: ADAS
- 2. Automated scientific discovery: AI Scientist
- 3. Code as an agent genome: AlphaEvolve
- 4. Failure modes and risks of open-ended evolution
- Summary
- Exercises
- References

### 08-search-deep-research.ipynb — Search and deep-research agents

- 1. Search in program synthesis: AlphaCode
- 2. AlphaCode 2: from sampling to filtering
- 3. Agent search-enhanced reasoning: Search-o1
- 4. Deep-research workflow
- Summary
- Exercises
- References

### 09-post-training-evolution.ipynb — Post-training evolution: from chatbot to agent

- 1. What post-training is
- 2. SFT and RLHF: the chatbot era
- 3. Agent post-training: tools, execution, feedback
- 4. Evolution roadmap
- Summary
- Exercises
- References

---

## Part 3 · Agent Engineering — L10, L11, L14

### 10-swe-agents.ipynb — Software engineering agents

- 1. Test-time compute on code tasks: CodeMonkeys
- 2. Letting LLMs write efficient kernels: KernelBench
- 3. Agent–system interface design
- 4. Building a SWE-Agent loop
- Summary
- Exercises
- References

### 11-agent-memory.ipynb — Agent memory

- 1. Why memory matters
- 2. MemGPT: LLMs as operating systems
- 3. Cartridges: self-study of long-context representations
- 4. CacheBlend: KV-cache reuse for RAG
- 5. Engineering practice for memory systems
- Summary
- Exercises
- References

### 14-agent-evaluation.ipynb — Agent evaluation and long-horizon tasks

- 1. Why evaluating agents is hard
- 2. Long-horizon evaluation: Measuring Long Tasks
- 3. Real economically valuable tasks: GDPval
- 4. Deep-research evaluation: DeepScholar-Bench
- 5. Building an evaluation harness
- Summary
- Exercises
- References

---

## Part 4 · Frontiers — L12, L13, L15–L17

### 12-llm-reasoning.ipynb — LLM reasoning

- 1. Where reasoning ability comes from
- 2. Chain-of-Thought and self-consistency
- 3. Emergence of reasoning
- 4. Development of reasoning models
- Summary
- Exercises
- References

### 13-alphaproof-math.ipynb — Mathematical reasoning agents

- 1. Mathematics as a test of reasoning
- 2. AlphaGeometry: neural–symbolic geometry
- 3. AlphaProof: RL for formal proofs
- 4. IMO gold: Gemini's path
- Summary
- Exercises
- References

### 15-autonomy-agents.ipynb — Building autonomous agents

- 1. From demos to autonomy
- 2. Reliability: error detection and recovery
- 3. Oversight and trust boundaries
- 4. Open questions
- Summary
- Exercises
- References

### 16-multimodal-robotics.ipynb — Multimodal robotics agents

- 1. Embodied intelligence and VLAs
- 2. Vision–language–action models
- 3. Collecting robot data
- 4. Feedback from learning in the physical world
- Summary
- Exercises
- References

### 17-future-research.ipynb — Future research directions

- 1. Limits of current systems
- 2. Open-problem list
- 3. Possible research paths
- 4. How to participate
- Summary
- Exercises
- References

---

## Course numbering

Lectures are numbered continuously in learning order as L1–L17. Notebook filenames, study-note directories, and the online reader use the same numbers.

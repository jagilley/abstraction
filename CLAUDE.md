# CLAUDE.md

This is a research monorepo consolidating several projects.

## Structure

- `experiments/` - All experiments
    - `experiments/a2a_forward` is our main experiment for training forward self-models for decomposing activations
    - `experiments/rhm` is a controlled setup for generating language-like data with a known data generating process
- `reading/` — Papers, notes, and reference material (shared across projects)
- `ideas/` — Research ideas and hypotheses (shared)
- `beliefs/` — Crystallized beliefs about how things work (shared)
- `conversations/` — Conversations with AI assistants and collaborators (shared)
- `fer/` — Fractured Entangled Representations project (JAX/Flax, conda env: `fer`)
- `glp/` — Generative Latent Prior project (PyTorch, conda env: `glp`)
- `manim-videos/` — Animated video explainers (Manim CE, conda env: `manim`)
- `org/` — Org-mode files for research and application tracking
- `applications/` — Job/fellowship applications
- `projects/` — Cross-project progress logs

Each project subdirectory has its own CLAUDE.md with setup instructions and conventions.

## Research taste
- We prefer seeking the fundamental causal structure over scoring high on the test.
- We prefer solutions that generalize across domains.

## Good practices
- We must keep all prior results reproducible. You're welcome to edit the code used in prior experiments, as long as you keep it backwards-compatible (within reason). If we're making too many backwards-compatible modifications, it's probably time to copy things over to a new folder/file. Keeping folders and files semantically meaningful is more important than not duplicating code.
- Document everything you do to the filesystem. If you're not sure where it should go, ask.
- Controlling variables is absolutely essential for good science.
- Please always discuss results with me before writing a new README. These things can be difficult to interpret sometimes.
- Don't get discouraged. If something didn't work, there's a reason for it, and we should understand what that reason is before we update our priors on why we wanted to try that thing in the first place.
- We should edit this doc as frequently as possible to persist learnings.

## Experiment conventions
- Each experiment gets its own folder in the project's `experiments/` directory.
- Put the `train.sh` for an experiment inside that experiment's directory.
- In `train.sh`, capture output and write it to disk for future debugging.
- Write a `README.md` in each experiment directory after results come back, so we have the complete picture. Include a pointer to the prior experiment's README so agents can reference it.
- Running quick tests to ensure code works before kicking off longer training jobs is usually a good idea.

## Shared directories

Reading, ideas, beliefs, and conversations are shared across projects. Experiments can live in project directories (`fer/experiments/`, `glp/experiments/`), but may be consolidated later.

## Conversation transcripts
- Claude Code transcripts are symlinked into `conversations/claude-code-transcripts/`.
- Before parsing prior conversations, run `python3 conversations/parse_transcripts.py` to generate `INDEX.md` and `readable/` markdowns from the raw JSONL files.
- Read `conversations/claude-code-transcripts/INDEX.md` first to find the relevant session, then read the specific `readable/<session>.md` file.
- Be sure to run the `conversations/parse_transcripts.py` script prior to reading the file, even if the file already exists. If you don't, you may read an out-of-date version of the convo.
- To get the id of the current Claude Code conversation, you can run `echo $CLAUDE_CODE_SESSION_ID`

## Notes
- For long-running e.g. training jobs (long running = anything that takes more than 2 mins), please follow this procedure to manage things:
    1. Kick off the job, make sure it runs, etc. You can do this by using a detached Modal run with a 2-minute timeout before the process is auto-backgrounded by the Claude Code harness.
    2. Then, don't monitor any further, halt your work, and just wait for the background notification that the job has completed. Under no circumstances should you read logs in consecutive tool calls as a means of waiting for the run to finish.
- Please don't use Explore subagents. These use Claude Haiku, which is too small of a model to be useful. Opt for Sonnet or Opus explore agents/subagents instead if you like, but for short tasks often nothing beats just looking at the data/code yourself.
- For reading background/context files, please also avoid using subagents. Consuming the relevant context directly is often load-bearing for interpreting results properly. Subagents, especially those that run smaller models, risk misinterpreting results given a lack of context.

## Quotes

- "If an explanation is long, there's a high chance that it's wrong" -- Ilya Sutskever
- "human collaboration is a superintelligence technology" -- Ilya Sutskever
- "It's not helpful to think of problems as of 'hard'. It's better to think that we merely don't know how to solve them yet." -- Ilya Sutskever
- "Creativity is an inverse problem" -- Ilya Sutskever
- "Creativity = novelty + value" -- Ilya Sutskever
- "psychology should become more and more applicable to AI as it gets smarter" -- Ilya Sutskever
- "The art of doing mathematics consists in finding that special case which contains all the germs of generality." -- David Hilbert
- "The whole idea of Science is, simply, reflective reasoning about a more reliable process for making the contents of your mind mirror the contents of the world." -- Eliezer Yudkowsky

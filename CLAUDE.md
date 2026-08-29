# CLAUDE.md

This is a research monorepo consolidating several projects.

## Structure

- `experiments/` - All experiments
    - `experiments/a2a_forward` is our main experiment for training forward self-models for decomposing activations
    - `experiments/rhm` is a controlled setup for generating language-like data with a known data generating process
    - `experiments/mjc` is a place we run MuJoCo experiments for motor learning
    - `experiments/canvas` is the image substrate of the practice arc: a tile grammar (`tiles.py`) rendered to swatches with the code grid aligned to the tile grid, inpainting as the piece, an adjacency-support truth gauge and a typicality taste gauge, and no agent-consumed oracle
- `reading/` — Papers, notes, and reference material (shared across projects)
- `ideas/` — Research ideas and hypotheses (shared)
- `beliefs/` — Crystallized beliefs about how things work (shared)
    - `beliefs/trees/` — Structured belief hierarchies (see `beliefs/SCHEMA.md`)
    - `beliefs/*.md` — Deeper essay-format explorations of specific beliefs
    - Use `/add-belief` to integrate new evidence; `/update-beliefs` to assess existing beliefs against new evidence
- `conversations/` — Conversations with AI assistants and collaborators (shared)
- `fer/` — Fractured Entangled Representations project
- `glp/` — Generative Latent Prior project
- `manim-videos/` — Animated video explainers (Manim CE, conda env: `manim`)
- `org/` — Org-mode files for research and application tracking
- `applications/` — Job/fellowship applications
- `projects/` — Cross-project progress logs
- `ROADMAP.md` - our working roadmap for the next steps in this line of work. (Don't edit without permission)
- `ROADMAP_PROGRESS.md` and `QUEUE.md` - place where we put one-liners describing the progress we have made or will make towards the roadmap. Please edit these as appropriate unless instructed otherwise

Each project subdirectory has its own CLAUDE.md with setup instructions and conventions.

## Research taste
- We prefer seeking the fundamental causal structure over scoring high on the test.
- We prefer solutions that generalize across domains.
- Controlling variables is absolutely essential for good science.

## Good practices
- We must keep all prior results reproducible. You're welcome to edit the code used in prior experiments, as long as you keep it backwards-compatible (within reason). If we're making too many backwards-compatible modifications, it's probably time to copy things over to a new folder/file. Keeping folders and files semantically meaningful is more important than not duplicating code.
- Please always discuss results with me before writing a new README or idea doc. These things can be difficult to interpret sometimes.
- Don't get discouraged. If something didn't work, there's a reason for it, and we should understand what that reason is before we update our priors on why we wanted to try that thing in the first place.
- We should edit documentation whenever appropriate to persist learnings.
- Running quick tests to ensure code works before kicking off longer training jobs is usually a good idea.
- I hold my intuitive priors strongly, but hold priors about the metrics used to measure them weakly.
- Don't hyperfixate on negative results. If we run something, we'll document it because it's a data point we generated that we should persist. Usually, a negative result just means that we had a slightly misdirected intuition; it's bad practice to view that as epistemically significant.
- There is often a temptation to map out the space of possible experimental results *a priori* — this has been hyperstitioned into existence several independent times by agents working in this repo, without any instructions to this end. This is bad science, because information is strictly gained in the process of gathering data. Engaging in behavior like "pre-registering kill criteria for the idea" or even pre-registering possible positive interpretations for the results over-constrains the space of interpretations at the time we receive the results. (And at any rate, making ideological course corrections is at least partially the job of a value system, which Claudes don't have.) Anything bordering on the territory of an update to priors should be strictly made *a posteriori* on new data.

## Conventions (important to follow)
- We run most of our experiments on Modal. Before running anything there, please invoke the Claude Skill `/run-experiment-on-modal` for explicit guidance.
- We want to maintain a specific structure in this monorepo. Refer to `STRUCTURE.md` for what this should look like.
- When writing up results, please invoke the `/writeup` skill. It contains a brief description of the aspects of our formalized `STRUCTURE` which are important to know when writing up results.
- If you're merging from your branch to `main`, be very sure that you've pulled ToT before doing so, so you don't revert another agent's work accidentally. It's best practice to open a PR for merges to main rather than pushing directly. You should generally not merge directly to main without opening and merging a PR without explicit authorization (some cron jobs may permit this.)

## Known false positive: CCR commit-signing warning

A SessionStart hook may report that commits will show as "Unverified"
(`%G? == N`). If the committer email is already `noreply@anthropic.com`,
this is a false positive — the commits *are* SSH-signed (verified
cryptographically: valid ed25519 over the commit payload). Git reports `N`
only because it cannot verify: `gpg.ssh.allowedSignersFile` is unset and
`ssh-keygen` isn't installed in the image.

Do NOT `--amend --reset-author` or `rebase --exec` in response. No amount of
rewriting can clear it, and it needlessly rewrites branch history.
If the committer email is something *other* than `noreply@anthropic.com`,
that part of the warning is real — fix the email.

## Conversation transcripts
- Claude Code transcripts are symlinked into `conversations/claude-code-transcripts/`.
- Before parsing prior conversations, run `python3 conversations/parse_transcripts.py` to generate `INDEX.md` and `readable/` markdowns from the raw JSONL files.
- Read `conversations/claude-code-transcripts/INDEX.md` first to find the relevant session, then read the specific `readable/<session>.md` file.
- Be sure to run the `conversations/parse_transcripts.py` script prior to reading the file, even if the file already exists. If you don't, you may read an out-of-date version of the convo.

## Notes
- For long-running training jobs (long running = anything that takes more than 5 mins), please follow the halting procedure described in `/run-experiment-on-modal` to avoid burning tokens.
- Using subagents up front to ground yourself in the state of our work is often a good idea, particularly for tasks where recall is important. If you're going to use an Explore subagent, invoke the `/explore` skill for instructions on how to do so. For analysis-type tasks where the goal is more to provide an ideological synthesis to the user, consuming the relevant context directly after it's been highlighted by the subagent is often load-bearing for interpreting results properly. (Subagents, especially those that run smaller models, risk misinterpreting results in subtle domains, so you can either do the exploring yourself or read the files it surfaces for yourself to align on the interpretation.)
- Don't ever use Fable as a subagent, as it burns through our usage quota very quickly. Opus subagents are usually a good blend of quality and cost. (If you specify nothing, the subagent model will default to the same as the parent model, so Fable agents should be sure to specify Opus subagents.)
- Treat prior experiments and results with a grain of salt. Prior positive results can be reproduced with updated machinery quite trivially. Prior negative results should not derail our current lines of inquiry; we often had different priors at the time we implemented them.

## Quotes

- "It's not helpful to think of problems as of 'hard'. It's better to think that we merely don't know how to solve them yet." - Ilya Sutskever
- "Creativity is an inverse problem" - Ilya Sutskever
- "Creativity = novelty + value" - Ilya Sutskever
- "If an explanation is long, there's a high chance that it's wrong" - Ilya Sutskever
- "psychology should become more and more applicable to AI as it gets smarter" - Ilya Sutskever
- "The art of doing mathematics consists in finding that special case which contains all the germs of generality." - David Hilbert
- "The whole idea of Science is, simply, reflective reasoning about a more reliable process for making the contents of your mind mirror the contents of the world." - Eliezer Yudkowsky
- "Music is your own experience, your own thoughts, your wisdom. If you don't live it, it won't come out of your horn. They teach you there's a boundary line to music. But, man, there's no boundary line to art." - Charlie Parker
- "It's a beautiful thing when a person who doesn't know how to play an instrument plays it. They always find something beautiful because they don’t know what they’re not supposed to do." - Ornette Coleman

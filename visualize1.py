# render_workflow.py
# Creates a clear LR graph of your LangGraph workflow with clusters and readable conditional edges.
# No imports from your project; pure Graphviz.

from graphviz import Digraph

g = Digraph("workflow", filename="workflow_graph", format="png")
g.attr(rankdir="LR", splines="ortho", nodesep="0.45", ranksep="0.6", fontname="Helvetica")

# ------- Common styles -------
node_core = {"shape": "box", "style": "rounded,filled", "fillcolor": "#eef5ff", "color": "#7aa2ff"}
node_impr = {"shape": "box", "style": "rounded,filled", "fillcolor": "#eefbea", "color": "#73c87a"}
node_step = {"shape": "box", "style": "rounded", "color": "#444"}
edge_cond = {"color": "#d17a00", "fontcolor": "#d17a00", "penwidth": "1.6"}
edge_retry = {"color": "#3e86ff", "fontcolor": "#3e86ff", "penwidth": "1.4"}

# ------- Start / End -------
g.node("START", shape="ellipse", style="filled", fillcolor="#ffffff", color="#777")
g.node("END", shape="ellipse", style="filled", fillcolor="#ffffff", color="#777")

# ================= Core Generation Path =================
with g.subgraph(name="cluster_core") as core:
    core.attr(label="Core Generation Path", color="#bcd3ff", style="rounded", bgcolor="#f7fbff", fontsize="12")

    for n in [
        "select_tools",
        "llm_tool_orchestration",
        "collect_tool_outputs",
        "retrieve_few_shots",
        "generate_intermediate_context",
        "create_prompt",
        "generate_code",
        "execute_code",
        "build_manifest",
        "reflect_code",
        "finalize",
    ]:
        # visually highlight the two “hubs”
        if n in {"execute_code", "build_manifest"}:
            core.node(n, **node_core)
        else:
            core.node(n, **node_step)

    # straight core edges (linear happy path up to execute_code)
    core.edge("START", "select_tools")
    core.edge("select_tools", "llm_tool_orchestration")
    core.edge("llm_tool_orchestration", "collect_tool_outputs")
    core.edge("collect_tool_outputs", "retrieve_few_shots")
    core.edge("retrieve_few_shots", "generate_intermediate_context")
    core.edge("generate_intermediate_context", "create_prompt")
    core.edge("create_prompt", "generate_code")
    core.edge("generate_code", "execute_code")

    # retry loop: reflect_code → execute_code
    core.edge("reflect_code", "execute_code", **edge_retry, xlabel=" retry")

    # finalize → END
    core.edge("finalize", "END")

# ================= Improvement Loop =================
with g.subgraph(name="cluster_improve") as imp:
    imp.attr(label="Improvement Loop", color="#a8e3ad", style="rounded", bgcolor="#f4fff4", fontsize="12")

    for n in ["eval_report", "tavily_research", "extra_tool_exec", "delta_code_gen"]:
        imp.node(n, **node_impr)

    # linear flow inside improvement loop
    imp.edge("eval_report", "tavily_research")
    imp.edge("tavily_research", "extra_tool_exec")
    imp.edge("extra_tool_exec", "delta_code_gen")

# ================= Cross-Cluster & Conditional Routes =================

# delta_code_gen → execute_code (back into core)
g.edge("delta_code_gen", "execute_code", ltail="cluster_improve", lhead="cluster_core",
       color="#666", penwidth="1.4", minlen="2")

# Conditional routes from execute_code
# execute_code → build_manifest  (success)
g.edge("execute_code", "build_manifest",
       xlabel=" post_execute: build_manifest", **edge_cond)

# execute_code → reflect_code   (failure; retry)
g.edge("execute_code", "reflect_code",
       xlabel=" post_execute: reflect_code", **edge_retry, minlen="2")

# execute_code → finalize       (terminal failure)
g.edge("execute_code", "finalize",
       xlabel=" post_execute: finalize", color="#cc5a5a", fontcolor="#cc5a5a", penwidth="1.6", minlen="2")

# Conditional routes from build_manifest
# build_manifest → eval_report  (keep improving)
g.edge("build_manifest", "eval_report",
       xlabel=" post_manifest: eval_report", **edge_cond,
       ltail="cluster_core", lhead="cluster_improve", minlen="3", constraint="false")

# build_manifest → finalize     (we’re done)
g.edge("build_manifest", "finalize",
       xlabel=" post_manifest: finalize", color="#cc5a5a", fontcolor="#cc5a5a",
       penwidth="1.6", minlen="2")

# Optional: keep START at far left and END at far right in same rank rows
g.attr(rank="same")
# (No explicit same-rank blocks needed because rankdir=LR and we control long edges with constraint/minlen.)

# Render
outpath = g.render(cleanup=True)
print(f"Wrote {outpath}")

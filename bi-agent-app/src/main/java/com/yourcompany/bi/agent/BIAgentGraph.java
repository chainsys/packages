package com.yourcompany.bi.agent;

import com.yourcompany.bi.agent.nodes.SqlExecutorNode;
import com.yourcompany.bi.agent.nodes.SqlGeneratorNode;
import com.yourcompany.bi.agent.nodes.VizRecommenderNode;
import java.lang.reflect.Method;
import java.util.function.Consumer;
import jakarta.enterprise.context.ApplicationScoped;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Orchestrates BI flow. Uses LangGraph4j if present, otherwise falls back to deterministic pipeline.
 */
@ApplicationScoped
public class BIAgentGraph {

    private static final Logger log = LoggerFactory.getLogger(BIAgentGraph.class);

    private final SqlGeneratorNode sqlGeneratorNode;
    private final SqlExecutorNode sqlExecutorNode;
    private final VizRecommenderNode vizRecommenderNode;

    public BIAgentGraph(
            SqlGeneratorNode sqlGeneratorNode,
            SqlExecutorNode sqlExecutorNode,
            VizRecommenderNode vizRecommenderNode) {
        this.sqlGeneratorNode = sqlGeneratorNode;
        this.sqlExecutorNode = sqlExecutorNode;
        this.vizRecommenderNode = vizRecommenderNode;
    }

    public AgentState run(AgentState state) {
        if (!runWithLangGraph4j(state)) {
            runSequential(state);
        }
        return state;
    }

    private void runSequential(AgentState state) {
        sqlGeneratorNode.apply(state);
        sqlExecutorNode.apply(state);
        vizRecommenderNode.apply(state);
    }

    /**
     * Reflective invocation keeps this class resilient across LangGraph4j beta API changes.
     */
    private boolean runWithLangGraph4j(AgentState state) {
        try {
            Class<?> graphClass = Class.forName("org.bsc.langgraph4j.StateGraph");
            Object graph = graphClass.getConstructor().newInstance();

            Method addNode = graphClass.getMethod("addNode", String.class, Consumer.class);
            Method addEdge = graphClass.getMethod("addEdge", String.class, String.class);
            Method setEntryPoint = graphClass.getMethod("setEntryPoint", String.class);
            Method setFinishPoint = graphClass.getMethod("setFinishPoint", String.class);
            Method compile = graphClass.getMethod("compile");

            addNode.invoke(graph, "sqlGenerate", (Consumer<AgentState>) sqlGeneratorNode::apply);
            addNode.invoke(graph, "sqlExecute", (Consumer<AgentState>) sqlExecutorNode::apply);
            addNode.invoke(graph, "vizRecommend", (Consumer<AgentState>) vizRecommenderNode::apply);
            addEdge.invoke(graph, "sqlGenerate", "sqlExecute");
            addEdge.invoke(graph, "sqlExecute", "vizRecommend");
            setEntryPoint.invoke(graph, "sqlGenerate");
            setFinishPoint.invoke(graph, "vizRecommend");

            Object compiledGraph = compile.invoke(graph);
            Method invoke = compiledGraph.getClass().getMethod("invoke", Object.class);
            invoke.invoke(compiledGraph, state);
            return true;
        } catch (Exception ex) {
            log.debug("LangGraph4j unavailable or API mismatch; using sequential pipeline. Cause: {}", ex.getMessage());
            return false;
        }
    }
}

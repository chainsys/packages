package com.yourcompany.bi.agent;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.yourcompany.bi.agent.nodes.SqlExecutorNode;
import com.yourcompany.bi.agent.nodes.SqlGeneratorNode;
import com.yourcompany.bi.agent.nodes.VizRecommenderNode;
import com.yourcompany.bi.ai.AiServiceFactory;
import com.yourcompany.bi.database.DbSchemaService;
import com.yourcompany.bi.database.SqlRunner;
import dev.langchain4j.model.chat.ChatLanguageModel;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Produces;

@ApplicationScoped
public class AgentConfig {

    @Produces
    @ApplicationScoped
    public ChatLanguageModel chatLanguageModel(AiServiceFactory aiServiceFactory) {
        return aiServiceFactory.createChatModel();
    }

    @Produces
    @ApplicationScoped
    public SqlGeneratorNode sqlGeneratorNode(ChatLanguageModel chatLanguageModel, DbSchemaService dbSchemaService) {
        return new SqlGeneratorNode(chatLanguageModel, dbSchemaService);
    }

    @Produces
    @ApplicationScoped
    public SqlExecutorNode sqlExecutorNode(SqlRunner sqlRunner) {
        return new SqlExecutorNode(sqlRunner);
    }

    @Produces
    @ApplicationScoped
    public VizRecommenderNode vizRecommenderNode(ChatLanguageModel chatLanguageModel, ObjectMapper objectMapper) {
        return new VizRecommenderNode(chatLanguageModel, objectMapper);
    }

    @Produces
    @ApplicationScoped
    public BIAgentGraph biAgentGraph(
            SqlGeneratorNode sqlGeneratorNode,
            SqlExecutorNode sqlExecutorNode,
            VizRecommenderNode vizRecommenderNode) {
        return new BIAgentGraph(sqlGeneratorNode, sqlExecutorNode, vizRecommenderNode);
    }
}

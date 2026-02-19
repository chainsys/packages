package com.yourcompany.bi.agent.nodes;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yourcompany.bi.agent.AgentState;
import com.yourcompany.bi.ai.PromptTemplates;
import dev.langchain4j.model.chat.ChatLanguageModel;

/**
 * Recommends the best chart type for result data.
 */
public class VizRecommenderNode {

    private final ChatLanguageModel chatLanguageModel;
    private final ObjectMapper objectMapper;

    public VizRecommenderNode(ChatLanguageModel chatLanguageModel, ObjectMapper objectMapper) {
        this.chatLanguageModel = chatLanguageModel;
        this.objectMapper = objectMapper;
    }

    public void apply(AgentState state) {
        if (!state.getErrors().isEmpty()) {
            return;
        }

        try {
            String prompt = PromptTemplates.buildVizRecommendationPrompt(state.getQueryResults());
            String json = chatLanguageModel.generate(prompt);
            JsonNode response = objectMapper.readTree(json);
            state.setVisualizationRecommendation(response.path("chartType").asText("table"));
            state.getMetadata().put("vizReason", response.path("reason").asText(""));
        } catch (Exception ex) {
            state.setVisualizationRecommendation("table");
            state.addError("Visualization recommendation fallback: " + ex.getMessage());
        }
    }
}

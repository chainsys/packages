package com.yourcompany.bi.agent;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Mutable state passed through graph nodes.
 */
public class AgentState {

    private String userQuestion;
    private String generatedSql;
    private List<Map<String, Object>> queryResults = new ArrayList<>();
    private String visualizationRecommendation;
    private final List<String> errors = new ArrayList<>();
    private final Map<String, Object> metadata = new LinkedHashMap<>();

    public String getUserQuestion() {
        return userQuestion;
    }

    public void setUserQuestion(String userQuestion) {
        this.userQuestion = userQuestion;
    }

    public String getGeneratedSql() {
        return generatedSql;
    }

    public void setGeneratedSql(String generatedSql) {
        this.generatedSql = generatedSql;
    }

    public List<Map<String, Object>> getQueryResults() {
        return queryResults;
    }

    public void setQueryResults(List<Map<String, Object>> queryResults) {
        this.queryResults = queryResults;
    }

    public String getVisualizationRecommendation() {
        return visualizationRecommendation;
    }

    public void setVisualizationRecommendation(String visualizationRecommendation) {
        this.visualizationRecommendation = visualizationRecommendation;
    }

    public List<String> getErrors() {
        return errors;
    }

    public void addError(String error) {
        this.errors.add(error);
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }
}

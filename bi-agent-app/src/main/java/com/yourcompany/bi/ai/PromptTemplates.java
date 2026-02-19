package com.yourcompany.bi.ai;

import java.util.List;
import java.util.Map;

/**
 * Centralized prompts used by BI agent nodes.
 */
public final class PromptTemplates {

    private PromptTemplates() {
    }

    public static String buildSqlGenerationPrompt(String question, String ddl) {
        return """
                You are a BI SQL assistant.
                Rules:
                1) Return exactly one read-only SQL statement.
                2) Use only tables/columns from the provided schema.
                3) Never emit DDL/DML.
                4) Prefer LIMIT 200 unless aggregation is requested.

                Schema:
                %s

                User question:
                %s
                """.formatted(ddl, question);
    }

    public static String buildVizRecommendationPrompt(List<Map<String, Object>> rows) {
        return """
                You are a BI visualization assistant.
                Recommend the best chart for the SQL result.
                Return strict JSON only with:
                {"chartType":"bar|line|pie|table","reason":"..."}

                Sample rows:
                %s
                """.formatted(rows.stream().limit(10).toList());
    }
}

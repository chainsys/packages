package com.yourcompany.bi.agent.nodes;

import com.yourcompany.bi.agent.AgentState;
import com.yourcompany.bi.ai.PromptTemplates;
import com.yourcompany.bi.database.DbSchemaService;
import dev.langchain4j.model.chat.ChatLanguageModel;

/**
 * Generates read-only SQL from natural language.
 */
public class SqlGeneratorNode {

    private final ChatLanguageModel chatLanguageModel;
    private final DbSchemaService dbSchemaService;

    public SqlGeneratorNode(ChatLanguageModel chatLanguageModel, DbSchemaService dbSchemaService) {
        this.chatLanguageModel = chatLanguageModel;
        this.dbSchemaService = dbSchemaService;
    }

    public void apply(AgentState state) {
        if (state.getUserQuestion() == null || state.getUserQuestion().isBlank()) {
            state.addError("User question is required to generate SQL.");
            return;
        }

        String prompt = PromptTemplates.buildSqlGenerationPrompt(state.getUserQuestion(), dbSchemaService.loadSchemaDdl());
        String sql = chatLanguageModel.generate(prompt).trim();
        state.setGeneratedSql(stripMarkdownCodeBlock(sql));
    }

    private String stripMarkdownCodeBlock(String sql) {
        return sql.replace("```sql", "").replace("```", "").trim();
    }
}

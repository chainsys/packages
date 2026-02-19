package com.yourcompany.bi.agent.nodes;

import com.yourcompany.bi.agent.AgentState;
import com.yourcompany.bi.database.SqlRunner;

/**
 * Executes generated SQL through JDBC.
 */
public class SqlExecutorNode {

    private final SqlRunner sqlRunner;

    public SqlExecutorNode(SqlRunner sqlRunner) {
        this.sqlRunner = sqlRunner;
    }

    public void apply(AgentState state) {
        if (!state.getErrors().isEmpty()) {
            return;
        }

        String sql = state.getGeneratedSql();
        if (sql == null || sql.isBlank()) {
            state.addError("Generated SQL is empty.");
            return;
        }

        try {
            state.setQueryResults(sqlRunner.run(sql));
        } catch (Exception ex) {
            state.addError("SQL execution failed: " + ex.getMessage());
        }
    }
}

package com.yourcompany.bi.database;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import javax.sql.DataSource;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

/**
 * JDBC-backed SQL runner for read-only BI queries.
 */
@ApplicationScoped
public class SqlRunner {

    @Inject
    DataSource dataSource;

    public List<Map<String, Object>> run(String sql) {
        if (!isReadOnly(sql)) {
            throw new IllegalArgumentException("Only SELECT/CTE queries are allowed.");
        }

        try (Connection connection = dataSource.getConnection();
             Statement statement = connection.createStatement();
             ResultSet rs = statement.executeQuery(sql)) {
            List<Map<String, Object>> rows = new ArrayList<>();
            int cols = rs.getMetaData().getColumnCount();
            while (rs.next()) {
                Map<String, Object> row = new LinkedHashMap<>();
                for (int i = 1; i <= cols; i++) {
                    row.put(rs.getMetaData().getColumnLabel(i), rs.getObject(i));
                }
                rows.add(row);
            }
            return rows;
        } catch (Exception ex) {
            throw new RuntimeException("SQL execution failed", ex);
        }
    }

    private boolean isReadOnly(String sql) {
        String normalized = sql.stripLeading().toLowerCase(Locale.ROOT);
        return normalized.startsWith("select") || normalized.startsWith("with");
    }
}

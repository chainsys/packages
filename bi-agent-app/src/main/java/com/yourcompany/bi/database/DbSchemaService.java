package com.yourcompany.bi.database;

import java.sql.Connection;
import java.sql.DatabaseMetaData;
import java.sql.ResultSet;
import java.util.ArrayList;
import java.util.List;
import javax.sql.DataSource;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

/**
 * Loads lightweight schema DDL for prompting.
 */
@ApplicationScoped
public class DbSchemaService {

    @Inject
    DataSource dataSource;

    public String loadSchemaDdl() {
        List<String> statements = new ArrayList<>();
        try (Connection connection = dataSource.getConnection()) {
            DatabaseMetaData meta = connection.getMetaData();
            try (ResultSet tables = meta.getTables(connection.getCatalog(), null, "%", new String[]{"TABLE"})) {
                while (tables.next()) {
                    String table = tables.getString("TABLE_NAME");
                    statements.add("CREATE TABLE " + table + " (" + loadColumns(meta, table) + ");");
                }
            }
        } catch (Exception ex) {
            return "-- Unable to introspect schema: " + ex.getMessage();
        }
        return String.join("\n", statements);
    }

    private String loadColumns(DatabaseMetaData meta, String table) throws Exception {
        List<String> columns = new ArrayList<>();
        try (ResultSet rs = meta.getColumns(null, null, table, "%")) {
            while (rs.next()) {
                columns.add(rs.getString("COLUMN_NAME") + " " + rs.getString("TYPE_NAME"));
            }
        }
        return String.join(", ", columns);
    }
}

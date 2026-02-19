package com.yourcompany.bi.web;

import com.yourcompany.bi.agent.AgentState;
import com.yourcompany.bi.agent.BIAgentGraph;
import com.yourcompany.bi.web.models.AgentRequest;
import com.yourcompany.bi.web.models.AgentResponse;
import jakarta.inject.Inject;
import jakarta.validation.Valid;
import jakarta.ws.rs.Consumes;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;

@Path("/api/bi-agent")
@Consumes(MediaType.APPLICATION_JSON)
@Produces(MediaType.APPLICATION_JSON)
public class AgentController {

    @Inject
    BIAgentGraph biAgentGraph;

    @POST
    @Path("/query")
    public AgentResponse run(@Valid AgentRequest request) {
        AgentState state = new AgentState();
        state.setUserQuestion(request.getQuestion());

        AgentState result = biAgentGraph.run(state);

        AgentResponse response = new AgentResponse();
        response.setSql(result.getGeneratedSql());
        response.setResults(result.getQueryResults());
        response.setVisualization(result.getVisualizationRecommendation());
        response.setMetadata(result.getMetadata());
        response.setErrors(result.getErrors());

        return response;
    }
}

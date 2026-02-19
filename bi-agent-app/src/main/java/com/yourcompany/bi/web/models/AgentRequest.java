package com.yourcompany.bi.web.models;

import jakarta.validation.constraints.NotBlank;

public class AgentRequest {

    @NotBlank
    private String question;

    public String getQuestion() {
        return question;
    }

    public void setQuestion(String question) {
        this.question = question;
    }
}

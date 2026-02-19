package com.yourcompany.bi.ai;

import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import jakarta.enterprise.context.ApplicationScoped;
import org.eclipse.microprofile.config.inject.ConfigProperty;

/**
 * Creates LangChain4j chat model instances.
 */
@ApplicationScoped
public class AiServiceFactory {

    @ConfigProperty(name = "bi.ai.api-key")
    String apiKey;

    @ConfigProperty(name = "bi.ai.model", defaultValue = "gpt-4o-mini")
    String model;

    @ConfigProperty(name = "bi.ai.temperature", defaultValue = "0.1")
    Double temperature;

    public ChatLanguageModel createChatModel() {
        return OpenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName(model)
                .temperature(temperature)
                .build();
    }
}

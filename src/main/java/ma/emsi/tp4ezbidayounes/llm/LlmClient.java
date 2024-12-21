package ma.emsi.tp4ezbidayounes.llm;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.rag.query.transformer.CompressingQueryTransformer;
import dev.langchain4j.rag.query.transformer.QueryTransformer;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import jakarta.enterprise.context.Dependent;

import java.io.Serializable;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

@Dependent
public class LlmClient implements Serializable {

    public interface Assistant {
        String chat(String prompt);
    }

    private static void configureLogger() {
        // Configure le logger sous-jacent (java.util.logging)
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE); // Ajuster niveau
        // Ajouter un handler pour la console pour faire afficher les logs
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    private String systemRole;
    private Assistant assistant;
    private ChatMemory chatMemory;

    private EmbeddingStore embeddingStoreML;
    private EmbeddingStore embeddingStoreRag;

    private QueryRouter router;

    private ContentRetriever retrieverML;
    private ContentRetriever retrieverRag;

    private RetrievalAugmentor augmentor;

    public LlmClient() {
        configureLogger();

        ChatLanguageModel modele = GoogleAiGeminiChatModel.builder()
                .apiKey(System.getenv("GEMINI_KEY"))
                .modelName("gemini-1.5-flash")
                .temperature(0.7)
                .logRequestsAndResponses(true)
                .build();


        Path pathRessource1, pathRessource2;
        try {
            String cheminRessource1 = "/ml.pdf";
            String cheminRessource2 = "/rag.pdf";

            URL fileUrl = LlmClient.class.getResource(cheminRessource1);
            pathRessource1 = Paths.get(fileUrl.toURI());

            fileUrl = LlmClient.class.getResource(cheminRessource2);
            pathRessource2 = Paths.get(fileUrl.toURI());

        } catch (URISyntaxException e) {
            throw new RuntimeException(e); // ou un autre traitement du problème...
        }


        DocumentParser documentParser = new ApacheTikaDocumentParser();

        Document document1 = FileSystemDocumentLoader.loadDocument(pathRessource1, documentParser);
        Document document2 = FileSystemDocumentLoader.loadDocument(pathRessource2, documentParser);

        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        this.embeddingStoreML = new InMemoryEmbeddingStore();
        this.embeddingStoreRag = new InMemoryEmbeddingStore();

        EmbeddingStoreIngestor ingestorML = EmbeddingStoreIngestor.builder()
                .embeddingStore(this.embeddingStoreML)
                .embeddingModel(embeddingModel)
                .documentSplitter(splitter)
                .build();

        EmbeddingStoreIngestor ingestorRag = EmbeddingStoreIngestor.builder()
                .embeddingStore(embeddingStoreRag)
                .embeddingModel(embeddingModel)
                .documentSplitter(splitter)
                .build();

        ingestorML.ingest(document1);
        ingestorRag.ingest(document2);

        this.retrieverML = EmbeddingStoreContentRetriever.builder()
                .embeddingModel(embeddingModel)
                .embeddingStore(embeddingStoreML)
                .maxResults(2)
                .minScore(0.5)
                .build();

        this.retrieverRag = EmbeddingStoreContentRetriever.builder()
                .embeddingModel(embeddingModel)
                .embeddingStore(embeddingStoreRag)
                .maxResults(2)
                .minScore(0.5)
                .build();

        class QueryRouterPersonalise implements QueryRouter {

            @Override
            public Collection<ContentRetriever> route(Query query) {
                String question = "Est-ce que la requête '" + query.text()
                        + "' porte sur l'IA ? "
                        + "Réponds seulement par 'oui', 'non', ou 'peut-être'.";
                String reponse = modele.generate(question);
                if (reponse.toLowerCase().contains("non")) {
                    // Pas de RAG
                    return Collections.emptyList();
                } else {
                    question = "Est_ce que la requête '" + query.text()
                            + "' porte sur le fine-tuning ou le RAG ? "
                            + "Réponds seulement par 'oui', 'non', ou 'peut-être'.";
                    reponse = modele.generate(question);
                    if(reponse.toLowerCase().contains("non")) {
                        return List.of(retrieverML);
                    }
                    else {
                        return List.of(retrieverRag);
                    }
                }
            }
        }


        this.router = new QueryRouterPersonalise();

        QueryTransformer transformer = CompressingQueryTransformer.builder()
                .chatLanguageModel(modele)
                .build();

        this.augmentor = DefaultRetrievalAugmentor.builder()
                .queryTransformer(transformer)
                .queryRouter(router)
                .build();

        this.chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        this.assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(modele)
                .chatMemory(chatMemory)
                .retrievalAugmentor(augmentor)
                .build();


    }

    public void setSystemRole(String systemRole) {
        if(!this.systemRole.equals(systemRole)) {
            this.systemRole = systemRole;
            this.chatMemory.clear();
            this.chatMemory.add(new SystemMessage(this.systemRole));
        }
    }

    public String envoyerMessage(String question) {

        return this.assistant.chat(question);
    }
}
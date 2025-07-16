// src/javascript/mockData.js

export const mockDatabase = {
    trendingTopics: [
        {
            id: "petroleo",
            nome: "Petróleo",
            sentimentoResumo: "Neutro",
            sentimentoDetalhado: { positivo: 30, neutro: 50, negativo: 20 },
            ultimaAtualizacao: "2025-05-27"
        },
        {
            id: "tecnologia",
            nome: "Tecnologia",
            sentimentoResumo: "Positivo",
            sentimentoDetalhado: { positivo: 65, neutro: 25, negativo: 10 },
            ultimaAtualizacao: "2025-05-28"
        },
        {
            id: "inflacao",
            nome: "Inflação",
            sentimentoResumo: "Negativo",
            sentimentoDetalhado: { positivo: 15, neutro: 35, negativo: 50 },
            ultimaAtualizacao: "2025-05-28"
        }
        // Adicione mais tópicos se desejar
    ],

    sentimentoMercadoHistorico: {
        labels: ["22/05", "23/05", "24/05", "25/05", "26/05", "27/05", "28/05"],
        datasets: [
            { label: 'Positivo(%)', data: [30, 35, 40, 38, 45, 50, 48], backgroundColor:'green', borderColor: 'green', fill: false, tension: 0.1 },
            { label: 'Negativo(%)', data: [25, 20, 18, 22, 15, 12, 14], backgroundColor:'red',borderColor: 'red', fill: false, tension: 0.1 },
            { label: 'Neutro(%)', data: [45, 45, 42, 40, 40, 38, 38], backgroundColor:'#4F94DA',borderColor: '#4F94DA', fill: false, tension: 0.1 }
        ]
    },

    politicosMencionados: [
        { nome: "João Silva", mencoesPercentual: 45, sentimentoMedioPredominante: "Neutro" },
        { nome: "Maria Oliveira", mencoesPercentual: 30, sentimentoMedioPredominante: "Positivo" },
        { nome: "Carlos Pereira", mencoesPercentual: 15, sentimentoMedioPredominante: "Negativo" }
    ],

    analiseTermoBuscado: { // Exemplo para "Petrobras"
        termoBuscado: "Petrobras",
        resumoSentimento: { positivo: 45, negativo: 25, neutro: 30 },
        indiceConfianca: { valor: 82, baseNoticias: 257, periodoDias: 30 },
        noticiasRelacionadas: [
            { id: "noticia1", titulo: "Petrobras anuncia novo recorde de produção", fonte: "Agência XYZ", data: "2025-05-28", sentimento: "positivo", resumo: "A estatal superou expectativas do mercado.", linkOriginal: "#" },
            { id: "noticia2", titulo: "Variação cambial e ações da Petrobras", fonte: "Jornal Financeiro", data: "2025-05-27", sentimento: "neutro", resumo: "Analistas debatem impacto do dólar nos papéis.", linkOriginal: "#" }
        ]
    },

    // NOVO: Array para as últimas notícias
    ultimasNoticias: [
        { id: "ult1", titulo: "Novas projeções econômicas para o segundo semestre", fonte: "Valor Econômico", data: "2025-05-28", sentimento: "neutro", resumo: "Analistas revisam expectativas para o PIB e inflação.", linkOriginal: "#" },
        { id: "ult2", titulo: "Setor de energia renovável atrai novos investimentos", fonte: "Exame", data: "2025-05-27", sentimento: "positivo", resumo: "Empresas anunciam aportes significativos em projetos solares.", linkOriginal: "#" },
        { id: "ult3", titulo: "Reforma tributária: desafios e oportunidades para empresas", fonte: "Estadão", data: "2025-05-26", sentimento: "negativo", resumo: "Debate no congresso segue intenso sobre os próximos passos.", linkOriginal: "#" },
        { id: "ult4", titulo: "Crescimento do e-commerce impulsiona logística no país", fonte: "Mercado & Consumo", data: "2025-05-25", sentimento: "positivo", resumo: "Demanda por armazéns e entregas rápidas em alta.", linkOriginal: "#" },
        { id: "ult5", titulo: "Preços de commodities agrícolas impactam mercado internacional", fonte: "Globo Rural", data: "2025-05-24", sentimento: "neutro", resumo: "Tendências globais e seus efeitos na produção local.", linkOriginal: "#" }
    ],

    getNoticias: function(termo) {
        if (termo && termo.toLowerCase().includes("petrobras")) {
            return this.analiseTermoBuscado.noticiasRelacionadas;
        }
        if (termo && termo.toLowerCase().includes("lula")) {
            return [
                { id: "noticiaL1", titulo: "Presidente Lula em cúpula internacional", fonte: "G1", data: "2025-05-28", sentimento: "neutro", resumo: "Discussões sobre acordos e meio ambiente.", linkOriginal: "#" },
                { id: "noticiaL2", titulo: "Novas medidas econômicas do governo Lula", fonte: "CNN Brasil", data: "2025-05-27", sentimento: "positivo", resumo: "Pacote visa estimular crescimento e empregos.", linkOriginal: "#" },
            ];
        }
        return [
            { id: "noticiaG1", titulo: "Mercado reage a dados da inflação e bolsa despenca", fonte: "InfoMoney", data: "2025-05-28", sentimento: "negativo", resumo: "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed in vulputate lacus. Fusce quis lectus in sem maximus lobortis sed quis urna. Donec et interdum arcu. Nam consequat bibendum quam, ut molestie turpis elementum id. Fusce ac turpis a ex sodales mattis. Ut mi massa, ultricies eget tellus sed, eleifend congue augue. Sed quam sapien, auctor ut luctus id, congue quis odio. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Cras faucibus pulvinar tortor egestas scelerisque. Aliquam quis orci et metus tempor porta placerat sed neque. Phasellus sit amet lacus blandit ex luctus volutpat. Nulla a arcu eget felis accumsan dapibus congue a odio. Quisque elementum aliquam velit accumsan tincidunt.", linkOriginal: "#" },
            { id: "noticiaG2", titulo: "Governo baixa tarifas internacionais", fonte: "Folha", data: "2025-05-27", sentimento: "positivo", resumo: "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed in vulputate lacus. Fusce quis lectus in sem maximus lobortis sed quis urna. Donec et interdum arcu. Nam consequat bibendum quam, ut molestie turpis elementum id. Fusce ac turpis a ex sodales mattis. Ut mi massa, ultricies eget tellus sed, eleifend congue augue. Sed quam sapien, auctor ut luctus id, congue quis odio. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Cras faucibus pulvinar tortor egestas scelerisque. Aliquam quis orci et metus tempor porta placerat sed neque. Phasellus sit amet lacus blandit ex luctus volutpat. Nulla a arcu eget felis accumsan dapibus congue a odio. Quisque elementum aliquam velit accumsan tincidunt.", linkOriginal: "#" },
            { id: "noticiaG2", titulo: "Congresso vota reforma na próxima semana", fonte: "Folha", data: "2025-05-27", sentimento: "neutro", resumo: "Expectativa de debate acalorado.", linkOriginal: "#" },
        ];
    }
};
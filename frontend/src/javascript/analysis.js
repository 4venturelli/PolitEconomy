const searchInputIndex = document.querySelector('.header-bottom-side .input-field')
const searchButtonIndex = document.querySelector('.header-bottom-side .search-button')
const searchField = document.querySelector('.search-target')
const selectTopicos = document.getElementById('select-topicos')

if (searchInputIndex) {
    searchInputIndex.addEventListener('keypress', (event) => {
      if (event.key === 'Enter') {
        const termo = searchInputIndex.value.trim()
        if(termo) {
          const campo = searchField.value
          window.location.href = `search.html?termo=${encodeURIComponent(termo)}&campo=${encodedComponent(campo)}`
        }
      }
    })
}
if (searchButtonIndex) {
  searchButtonIndex.addEventListener('click', () => {
    const termo =searchInputIndex.value.trim()
    if(termo) {
      const campo = searchField.value
      window.location.href = `search.html?termo=${encodeURIComponent(termo)}&campo=${encodeURIComponent(campo)}`
    }
  });
}

async function buscarTopicos(quantidade, secao='') {
  if(isNaN(parseInt(quantidade))) {
    console.log("ERRO! Deve ser especificado a quantidade de tópicos a serem buscados.")
    return null
  }
  let urlBase = `http://localhost:3000/api/em-alta?quantidade=${quantidade}`
  if(secao != '' && secao !== 'Economia e Política') {
    urlBase = urlBase.concat(`?secao=${encodeURIComponent(secao)}`)
  }
  let res = await fetch(urlBase)
  if(!res.ok) {
    console.log("Não foi possível carregar os dados.")
    return null
  }
  let topicos = await res.json()
  return topicos
}

async function buscarUltimasNoticias(secao) {
  let urlBase = `http://localhost:3000/api/ultimas-noticias`
  if(secao != '' && secao != 'Economia e Política') {
    urlBase = urlBase.concat(`?secao=${encodeURIComponent(secao)}`)
  }
  const res = await fetch(urlBase);
  if(!res.ok) {
    throw new Error("Ocorreu um erro ao buscar as últimas notícias.")
  }
  const data = await res.json();
  return data
}

function carregaTrendingTopics(topicos) {
  const topicsContainer = document.getElementById('trending-topics-container')
  topicsContainer.innerHTML = ''
  
  topicos.forEach((topico) => {
    const topicoCard = document.createElement('div')
    topicoCard.className = 'trend'
    let buttonClass = 'trending-button'
    let sentimentoClass = 'neutral'
    if(topico.mediaDeSentimento > 0.2) {
      sentimentoClass = 'positive'
    } else if (topico.sSentimento < -0.2) {
      sentimentoClass = 'negative'
    }
    topicoCard.innerHTML = `
      <button class="${buttonClass} ${sentimentoClass}">${topico._id}</button>
      <div class="mini-sentiment-chart"></div>
      <p>Quantidade de notícias: ${topico.quantidadeNoticias}</p>
      <p>Última atualização: ${new Date(topico.dataUltimaAtualizacao).toLocaleDateString('pt-BR')}</p>
    `

    const buttonElement = topicoCard.querySelector('button')
    buttonElement.addEventListener('click', () => {
      window.location.href = `search.html?termo=${encodeURIComponent(topico._id)}&campo=${encodeURIComponent('Tópico')}`
    })
    topicsContainer.appendChild(topicoCard)
  })
}

function carregaRakingDeTopicos(topicos) {
  if(topicos === null) {
    console.log("Não foi possível buscar os tópicos.")
  }
  const data = [{
      key: "Quantidade de Notícias",
      values: topicos.map(topico => ({
        label: topico._id,
        value: topico.quantidadeNoticias
      }))
    },
    {
      key: "Média de sentimento",
      values: topicos.map(topico => {
        let media = (topico.mediaDeSentimento === null) ? 0 : topico.mediaDeSentimento.toFixed(2)
        let dados = {
          label: topico._id,
          value: media
        }
        return dados
      })
    }
  ]

    nv.addGraph(function () {
      var chart = nv.models.multiBarHorizontalChart()
        .x(function (d) { return d.label })
        .y(function (d) { return d.value })
        .margin({ top: 30, right: 20, bottom: 50, left: 175 })
        .showValues(true)
        .tooltips(true)
        .showControls(false)
        .color(['#1f77b4', '#f15f5f'])

      chart.yAxis
        .tickFormat(d3.format('d'))

      d3.select('.trend-ranking svg')
        .datum(data)
        .transition().duration(500)
        .call(chart)

      nv.utils.windowResize(chart.update);
      return chart
    });
}

async function carregarUltimasNoticias() {
  const carouselTrack = document.getElementById('latest-news-carousel')
  try {
    const data = await buscarUltimasNoticias(selectTopicos.value)
    carouselTrack.innerHTML = '';
    if (!data || data.length === 0) {
        carouselTrack.innerHTML = '<p>Nenhuma notícia recente encontrada.</p>';
        console.log("carregarUltimasNoticias: Nenhuma notícia para renderizar.");
        return;
    }
    data.forEach(noticia => {
        const newsCard = document.createElement('div');
        newsCard.className = 'carousel-item';
        let sentimentClass = 'notice-neutral';
        if (noticia.sSentimento === 1) sentimentClass = 'notice-positive';
        else if (noticia.sSentimento === -1) sentimentClass = 'notice-negative';
        const nomeAutor = Array.isArray(noticia.autor) && noticia.autor.length > 0
            ? noticia.autor.map(a => a.nome).join(', ')
            : 'Desconhecido';
        newsCard.innerHTML = `
            <div class="${sentimentClass}"></div> 
            <div>
                <h4>${noticia.manchete}</h4>
                <p>Fonte: ${noticia.portal?.nome || 'Desconhecida'} - Autor: ${nomeAutor} - Data: ${new Date(noticia.data).toLocaleDateString('pt-BR')}</p>
                <p style="font-size:0.9em; color:#333;">${noticia.lide || ''}</p>
                ${noticia._id && noticia._id !== "#" ? `<a href="${noticia._id}" target="_blank" style="font-size:0.9em;">Leia mais</a>` : ''}
            </div>
        `;
        carouselTrack.appendChild(newsCard);
    });
  } catch (err) {
    console.error("carregarUltimasNoticias: Erro ao buscar notícias:", err);
    carouselTrack.innerHTML = '<p>Não foi possível carregar as últimas notícias.</p>';
  }
}

async function buscaSentimentos(secao) {
  console.log(secao)
  let urlBase = `http://localhost:3000/api/sentimentos-gerais`
  if(typeof secao === 'string' && secao !== '' && secao !== 'Economia e Política') {
    urlBase = urlBase.concat(`?secao=${encodeURIComponent(secao)}`)
  }
  console.log(urlBase)
  let res =  await fetch(urlBase)
  if(!res.ok) {
    console.log("Não foi possível obter os dados sobre os sentimentos diários.")
    return null
  }
  let sentimentos = await res.json()
  return sentimentos
}

async function buscaProporcaoPortais(secao) {
  let urlBase = `http://localhost:3000/api/sentimentos-portais`
  if(typeof secao === 'string' && secao !== '' && secao !== 'Economia e Política') {
    urlBase = urlBase.concat(`?secao=${encodeURIComponent(secao)}`)
  }
  console.log('')
  let res =  await fetch(urlBase)
  if(!res.ok) {
    console.log("Não foi possível obter os dados sobre os sentimentos diários.")
    return null
  }
  let sentimentos = await res.json()
  return sentimentos
}

async function carregarProporcaoPortais(secao = '') {
  let portais = await buscaProporcaoPortais(secao)
  if(portais == null) {
    return
  }
  let dados = []
  let totalNoticias = 0
  for(let i = 0; i < portais.length; i++) {
    totalNoticias += portais[i].total
  }

  portais.forEach((portal) => {
    if(portal.total === 0) return
    dados.push({
      label: portal._id, 
      value: (portal.total / totalNoticias) * 100
    })
  })
  nv.addGraph(function() {
  var chart = nv.models.pieChart()
      .x(function(d) { return d.label })
      .y(function(d) { return d.value })
      .showLabels(false)

    d3.select("#proporcao-portais svg")
        .datum(dados)
      .transition().duration(1200)
        .call(chart)

    return chart
  });

}

function formatarDadosGraficoDeSentimento(dadosBrutos) {
  let positivas = {
    key: "Positivas",
    values: []
  }
  
  let neutras = {
    key: "Neutras",
    values: []
  }
  
  let negativas = {
    key: "Negativas",
    values: []
  }
  
  dadosBrutos.forEach(function(dados) {
    const dia = (dados._id.dia) ? dados._id.dia : '01'
    const mesAnoDia = `${dados._id.ano}-${dados._id.mes.toString().padStart(2, '0')}-${dia.toString().padStart(2, '0')}`;
    const data = new Date(`${mesAnoDia}`)
    
    let total = dados.total || 1;
    
    positivas.values.push({
      x: data,
      y: (dados.positivas / total) * 100
    })
    
    neutras.values.push({
      x: data,
      y: (dados.neutras / total) * 100
    })
    
    negativas.values.push({
      x: data,
      y: (dados.negativas / total) * 100
    })
  })
  
  return [positivas, neutras, negativas]
}

async function carregarGraficoSentimento(secao = '') {
  let sentimentos = await buscaSentimentos(secao)
  if(sentimentos === null) {
    return
  }
  let dados = formatarDadosGraficoDeSentimento(sentimentos)

  nv.addGraph(function() {
    var chart = nv.models.lineChart()
      .color(['green', '#4F94DA', 'red'])
      .useInteractiveGuideline(true)

    chart.xAxis
      .axisLabel('Dias')
      .tickFormat(function(d) {
          const date = (d instanceof Date) ? d : new Date(d)
          return d3.time.format('%d %b')(date)
        });

    chart.yAxis
      .axisLabel('Porcentagem')
      .tickFormat(d3.format('.02f'))
      ;

    d3.select('#sentimento-noticias svg')
      .datum(dados)
      .transition().duration(500)
      .call(chart)
      ;

    nv.utils.windowResize(chart.update);

    return chart;
  });
}

document.addEventListener('DOMContentLoaded', async () => {
  const topicos = await buscarTopicos(20)
  if(topicos == null) return
  carregaTrendingTopics(topicos.slice(0, 3))
  carregaRakingDeTopicos(topicos)
  carregarUltimasNoticias()
  carregarGraficoSentimento()
  carregarProporcaoPortais()
})

selectTopicos.addEventListener('change', async () => {
  const secao = (selectTopicos.value !== 'Economia e Política') ? selectTopicos.value : '' 
  carregarUltimasNoticias()
  const topicos = await buscarTopicos(20, secao)
  carregaTrendingTopics(topicos.slice(0,3))
  carregaRakingDeTopicos(topicos)
  carregarGraficoSentimento(secao)
  carregarProporcaoPortais(secao)
})
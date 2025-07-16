import { renderizarNoticias, buscarTermo, buscaSentimentoPorTempo } from "./script.js";

const searchInputIndex = document.querySelector('.header-bottom-side .input-field');
const searchButtonIndex = document.querySelector('.header-bottom-side .search-button'); 
const searchField = document.querySelector('.search-target');
let paginaIndex = 1

if (searchInputIndex) {
    searchInputIndex.addEventListener('keypress', (event) => {
      if (event.key === 'Enter') {
        const termo = searchInputIndex.value
        if(termo) {
          let buscaParams = parametrosDeBusca(termo)
          paginaIndex = 1
          limpaCampoNoticia()
          atualizaURL(buscaParams)
          carregaNoticias(buscaParams)
          desenhaGraficoSentimentoTempo(buscaParams)
        }
      }
    })
}
if (searchButtonIndex) {
  searchButtonIndex.addEventListener('click', () => {
    const termo = searchInputIndex.value
    if(termo != '') {
      let buscaParams = parametrosDeBusca(termo)
      paginaIndex = 1
      limpaCampoNoticia()
      atualizaURL(buscaParams)
      carregaNoticias(buscaParams)
      desenhaGraficoSentimentoTempo(buscaParams)
    }
  });
}

document.addEventListener('DOMContentLoaded', async () =>  {
  const searchPageTitleElement = document.getElementById('search-page-title');
  if (searchPageTitleElement) {
    const params = new URLSearchParams(window.location.search);
    const buscaParams = {}
    for(let [key, value] of params) {
      buscaParams[key] = decodeURIComponent(value)
    }
    //carregarAnaliseTermo(buscaParams);
    const noticias = await buscarTermo(buscaParams)
    preencheNoticias(noticias, buscaParams.termo)
    preencheSelectPortais()
    desenhaGraficoSentimentoTempo(buscaParams)
   //await graficoSentimentoSearch()
  }
})

function parametrosDeBusca(termo) {
  const buscaParams = {termo: termo}
  let dataInicio = document.getElementById('dataInicio').value
  let dataLimite = document.getElementById('dataLimite').value
  let portal = document.getElementById('portalSelect').value
  const campo = searchField.value
  if(campo !== '') buscaParams.campo = campo
  if(dataInicio !== '') buscaParams.dataInicio = dataInicio
  if(dataLimite !== '') buscaParams.dataLimite = dataLimite
  if(portal !== '') buscaParams.portal = portal
  return buscaParams
}

function atualizaURL(buscaParams) {
  const currentUrl = new URL(window.location.href);
  for(let i in buscaParams) {
    if(i === 'pagina') continue
    currentUrl.searchParams.set(i, encodeURIComponent(buscaParams[i]))
  }
  window.history.pushState({path: currentUrl.href}, '', currentUrl.href) // Atualiza a URL sem recarregar
}

async function carregaPortais() {
  let res = await fetch(`http://localhost:3000/api/portais`)
  if(!res.ok) {
    console.log("Não foi possível recuperar os portais disponíveis")
  }

  let portais = await res.json()
  portais = portais.map(x => x._id)
  return portais
}

async function preencheSelectPortais() {
  let portais = await carregaPortais()
  let select = document.getElementById('portalSelect')
  portais.forEach((portal) => {
    let option = document.createElement('option')
    option.setAttribute('value', portal)
    option.innerHTML = portal 
    select.appendChild(option)
  })
}

async function carregaNoticias(buscaParams) {
  const sentimentos = await buscaSentimentoPorTempo(buscaParams)
  console.log(sentimentos[0])
  const noticias = await buscarTermo(buscaParams)
  console.log(buscaParams)
  preencheNoticias(noticias, buscaParams.termo)
}

function preencheNoticias(noticias, termo) {
  let titulo = document.getElementById('search-page-title')
  if(noticias === null) {
    titulo.innerHTML = 'Houve um erro ao realizar a busca.'
  }
  if(noticias.length === 0 && paginaIndex == 1) {
    titulo.innerHTML = 'Nenhuma notícia encontrada.'
    return;
  }
  if(noticias === null && paginaIndex > 1) {
    return; //desabilitar o botão depois
  }
  titulo.innerHTML = `Resultado para: <span>${termo}</span>`
  renderizarNoticias(noticias)
}

function limpaCampoNoticia() {
  const noticesContainer = document.querySelector('.notices');
  noticesContainer.innerHTML = ''
}

function transformaDadosSentimentoPorTempo(dados) {
  const streams = {
    positivas: { key: "Positivas", values: [] },
    neutras: { key: "Neutras", values: [] },
    negativas: { key: "Negativas", values: [] }
  };

  dados.forEach(dados => {
    const dia = (dados._id.dia) ? dados._id.dia : '01'
    const mesAnoDia = `${dados._id.ano}-${dados._id.mes.toString().padStart(2, '0')}-${dia.toString().padStart(2, '0')}`;
    const data = new Date(`${mesAnoDia}`)
    streams.positivas.values.push({ x: data, y: dados.positivas });
    streams.neutras.values.push({ x: data, y: dados.neutras });
    streams.negativas.values.push({ x: data, y: dados.negativas });
  });

  return Object.values(streams);
}

async function desenhaGraficoSentimentoTempo(buscaParams) {
  const sentimento = await buscaSentimentoPorTempo(buscaParams)
  if(sentimento === null) return
  const dados = transformaDadosSentimentoPorTempo(sentimento)
  console.log(dados)

  nv.addGraph(function() {
    var chart = nv.models.multiBarChart()
                  .color(['green','#4F94DA','red'])

    chart.xAxis
        .tickFormat(function(d) {
          const date = (d instanceof Date) ? d : new Date(d)
          return d3.time.format('%b %Y')(date)
        });

    chart.yAxis
        .tickFormat(d3.format(',.1f'));

    d3.select('#sentimentoTempo svg')
        .datum(dados)
        .transition().duration(500)
        .call(chart)
        

    nv.utils.windowResize(chart.update)

    return chart
  })
}

document.querySelector('.carregar-mais').addEventListener('click', async () => {
      const params = new URLSearchParams(window.location.search)
      const buscaParams = {}
      for(let [key, value] of params) {
          buscaParams[key] = value
      }
      paginaIndex++
      buscaParams['pagina'] = paginaIndex
      let data = await buscarTermo(buscaParams)
      console.log(data)
      if(data != null) {
          preencheNoticias(data, buscaParams.termo)
      }
  })


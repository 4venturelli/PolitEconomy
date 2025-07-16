const puppeteer = require('puppeteer')
const {conectaBD, desconectaBD, insereNoticias} = require("./BD_handler")

async function coletaDadosCamaraDep(pagina, link) {
  await pagina.goto(link)
  const noticia = await pagina.evaluate(() => {
    let dados = {
      portal: {nome:"Portal da Câmara dos Deputados"},
      _id: window.location.href
    }

    // Manchete
    let manchete = document.querySelector("h1.g-artigo__titulo")
    if(manchete) {
      dados.manchete = manchete.textContent
    } else {
      return null
    }

    // Lide
    let lide = document.querySelector("p.g-artigo__descricao")
    if(lide) dados.lide = lide.textContent

    // Data
    let dataPublicacao = document.querySelector("p.g-artigo__data-hora")
    if(dataPublicacao) {
      dataPublicacao = dataPublicacao.textContent.trim()
      if(dataPublicacao.indexOf("•") >= 0) {
        dataPublicacao = dataPublicacao.slice(0, dataPublicacao.indexOf("•")).trim()
      }
      dataPublicacao = dataPublicacao.split(" - ")
      let dia = dataPublicacao[0]
      let hora = dataPublicacao[1]
      dia = dia.split("/")
      dia.reverse()
      dia = dia.join("-")
      let dataFormatada = `${dia}T${hora}-03:00`
      dados.data = dataFormatada
    } else {
      return null
    }

    // Autores
    let autores = document.querySelector("div.js-article-read-more p[style='font-size: 0.8rem; font-weight: 700;']")
    if(autores) {
      autores = autores.innerHTML
      autores = autores.split("<br>")
      autores = autores.map(x => {
        let resultado = undefined
        if(x.search("Reportagem –") >= 0) {
          let index = x.search("Reportagem –")
          resultado = x.slice("Reportagem –".length + index + 1).trim()
        } else if (x.search("Edição –") >= 0) {
          let index = x.search("Edição –")
          resultado = x.slice("Edição –".length + index + 1).trim()
        } else if (x.search("Da Redação") >= 0) {
          resultado = "Da Redação"
        }
        if(typeof resultado !== "undefined") {
          let autor = {nome: resultado}
          return autor
        } else {
          return null
        }
      })
      autores = autores.filter(autor => autor !== null)
      if(autores.length > 0) {
        dados.autor = autores.filter(autor => autor !== null)
      }
    }

    return dados
  })
  if(noticia != null) {
    noticia.data = new Date(noticia.data)
    if((noticia.data instanceof Date) == false) noticia = null
  }
  return noticia
}

async function scrapCamaraDeputados(URL, secao) {
  const browser = await puppeteer.launch()
  const scrapingPage = await browser.newPage()
  const paginaPortal = await browser.newPage()
  const {db, client} = await conectaBD()
  let dict = []

  try {
    for (let pagina = 1; pagina < 60; pagina ++) {
      let deputadosURL = `${URL}${pagina}`
      await paginaPortal.bringToFront()
      await paginaPortal.goto(deputadosURL, { waitUntil: "domcontentloaded" })

      const links = await paginaPortal.evaluate(() => {
        return Array.from(document.querySelectorAll("h3.g-chamada__titulo a")).map(x => x.getAttribute("href"))
      })
      
      await scrapingPage.bringToFront()
      for (let i = 0; i < links.length; i++) {
        let temp = await coletaDadosCamaraDep(scrapingPage, links[i])
        if(temp == null) continue

        temp.secao = secao
        dict.push(temp)
      }
    
      let inseridos = await insereNoticias(dict, db)
      if(inseridos < 0.5) {
        return null
      }
      dict = []
    }
    if(dict.length > 0) await insereNoticias(dict, db)
  } catch (err) {
    return
    console.error("Erro:", err)
  } finally {
    await desconectaBD(client)
    await scrapingPage.close()
    await browser.close()
  }
}

async function scrapingCamaraDeputados(){
  console.log('Coletando Portal da Câmara dos Deputados...')
  await scrapCamaraDeputados("https://www.camara.leg.br/noticias/ultimas?pagina=", "Política")
}

module.exports = {scrapingCamaraDeputados}


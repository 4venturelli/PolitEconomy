const puppeteer = require('puppeteer')
const { conectaBD, desconectaBD, insereNoticias } = require("./BD_handler")


async function coletaDadosTempo(pagina, link) {
  await pagina.goto(link, { waitUntil: "domcontentloaded" })
  const noticia = await pagina.evaluate(() => {
    const dados = {
      portal: {nome:"O Tempo"},
      _id: window.location.href,
    }

    // Manchete
    let manchete = document.querySelector("h1.cmp__title-title")
    if(manchete) dados.manchete = manchete.textContent.trim()
    else return null

    // Lide  
    let lide = document.querySelector("h2.cmp__title-bigode")
    if(lide) dados.lide = lide.textContent.trim()

    // Data
    let dataPublicacao = document.querySelector('.cmp__author-publication span')
    if(dataPublicacao){
      let meses = {
        janeiro: '01', fevereiro: '02', março: '03', abril: '04', maio: '05', junho: '06', julho: '07', agosto: '08', setembro: '09', outubro: '10', novembro: '11', dezembro: '12'
      };
      let [dataNova, horaNova] = dataPublicacao.textContent.split('|').map(p => p.trim());
      let data2 = dataNova.split(' '); 
    
      let dia = data2[0];
      let mesNome = data2[2];
      let ano = data2[4];

      let mesNumero = meses[mesNome.toLowerCase()];

      let dataFormatada = `${ano}-${mesNumero}-${dia.padStart(2, '0')}T${horaNova}:00`; 
      dados.data = dataFormatada
    } else {
      return null
    }

    // Autores
    let autoresTag = document.querySelector('.cmp__author-name span')
    if (autoresTag) {
      let autores = autoresTag.textContent.trim();
      if(autores.indexOf(' e ') >= 0) {
        autores = autores.split(' e ')
      } else {
        autores =  autores.split(/[,\|]/).map(a => a.trim()).filter(a => a.length > 0)
      }
      let array = autores
      dados.autor = array.map(x => {
        let autor = {nome: x}
        return autor
      })
    }

    // Tags
    let tema = document.querySelector("meta[name='keywords']")
    if(tema) {
      tema = tema.getAttribute('content').split(',')
      dados.assunto = tema.map(x => {
        let tag = {TAG: x}
        return tag
      })
    }

    return dados
  })
  if(noticia != null) {
    noticia.data = new Date(noticia.data)
    if((noticia.data instanceof Date) == false) noticia = null
  }
  return noticia
}


async function scrapTempo(URL, secao) {
  const browser = await puppeteer.launch({headless:true})
  const scrapingPage = await browser.newPage()
  const paginaPortal = await browser.newPage()

  const {db, client} = await conectaBD()
  let dict = []

  try {
    for (let pagina = 0; pagina <= 50; pagina++) {
      let tempoURL = `${URL}${pagina}`
      await paginaPortal.bringToFront()
      await paginaPortal.goto(tempoURL, { waitUntil: "domcontentloaded" })

      const links = await paginaPortal.evaluate(() => {
        return Array.from(document.querySelectorAll("a.list__link")).map(x => "https://www.otempo.com.br".concat(x.getAttribute("href")))
      })

      await scrapingPage.bringToFront()
      for (let i = 0; i < links.length; i++) {
        let temp = await coletaDadosTempo(scrapingPage, links[i])

        if(temp == null) continue;
        temp.secao = secao
        
        dict.push(temp)      
      }

      let inseridos = await insereNoticias(dict, db)
      if(inseridos < 0.5) {
        return null
      }
      dict = []
    }
    
  } catch (err) {
    return
    console.error("Erro:", err)
  } finally {
    await scrapingPage.close()
    await desconectaBD(client)
    await browser.close()
  }
}

async function scrapingTempo(){
  console.log('Coletando O Tempo...')
   await Promise.all([
    scrapTempo("https://www.otempo.com.br/politica/page/", "Política"),
    scrapTempo("https://www.otempo.com.br/economia/page/", "Economia")
   ])
}

module.exports = {scrapingTempo}
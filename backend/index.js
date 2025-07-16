require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose'); // acessar mongodb
const bcrypt = require('bcrypt'); // manusear senhas
const jwt = require('jsonwebtoken'); // passagem de token
const nodemailer = require('nodemailer') // enviar e-mails
const cron = require('node-cron') // agendar tarefas
const cors = require('cors');

const {
  assuntosEmAlta,
  ultimasNoticias,
  noticiasBuscaAvancada,
  portaisDisponiveis,
  sentimentoAtravesDoTempo,
  sentimentosGerais,
  sentimentosPortais,
} = require('./banco_de_dados/bancoSearch') 

const {
  processaFiltros
} = require('./filtros')

// Models
const { createProtocolErrorMessage } = require('puppeteer');
const User = require('./user'); // puxa modelo de como deve salvar o usuario no banco
const Noticia = require('./noticia'); 

const app = express();
const PORT = 3000;

// Middlewares
app.use(cors());
app.use(express.json());
app.use(express.static('public')); //adicionado para login e register
// precisa do cors pq o navegador acha que os dados estão vindo de fontes diferentes

// Habilita CORS para qualquer origem (durante o desenvolvimento)

app.get('/api/em-alta', async (req, res) => {
  let query = req.query, quantidade = 3, secao = ''
  if(Object.keys(query).length > 0)  {
    if(Object.hasOwnProperty.bind(query)('quantidade')) {
      quantidade = parseInt(query.quantidade)
      if(isNaN(quantidade)) {
        return res.status(404).send({error: 'Quantidade de tópicos inválida!'})
      }
    }
    if(Object.hasOwnProperty.bind(query)('secao')) {
      secao = query.secao
    }
  }
  let searchResult = await assuntosEmAlta(7, quantidade, secao)
  res.status(200).send(searchResult)
})

app.get('/api/em-alta/:dias', async (req, res) => {
  let reqDias = req.params.dias 
  let parsedDias = parseInt(reqDias)
  if(isNaN(parsedDias)) return res.status(404).send({error: 'ID inválido, deve ser um número.'})
  if(parsedDias < 0) return res.status(404).send({error: 'ID inválido, a quantidade de dias deve ser não negativo.'})
  if(parsedDias > 365) return res.status(404).send({error: 'O intervalo máximo para buscar os tópicos em alta é do período de 1 ano.'})
  let query = req.query, secao = '', quantidade = 3
  if(Object.keys(query).length > 0) {
    quantidade = parseInt(query.quantidade)
    if(isNaN(quantidade)) {
      return res.status(404).send({error: 'Quantidade de tópicos inválida!'})
    }
    if(Object.hasOwnProperty.bind(query)('secao')) {
      secao = query.secao
    }
  }
  let assuntos = await assuntosEmAlta(parsedDias, quantidade, secao)
  res.status(200).send(assuntos)
})


app.get('/api/assunto/:TAG', async (req, res) => {
  let tag = req.params.TAG
  let query = req.query
  let filtros
  try {
    filtros = processaFiltros(query)
  } catch(e) {
    res.status(404).send(e)
    return
  }
  if(filtros.campo === 'assunto.TAG') {
    tag = tag.toUpperCase()
  }

  let noticias = await noticiasBuscaAvancada(tag, filtros)
  res.status(200).send(noticias)
})

app.get('/api/ultimas-noticias', async (req, res) => { 
  let query = req.query, secao = ''
  if(Object.keys(query).length > 0) {
    if(Object.hasOwnProperty.bind(query)('secao'))
      secao = query.secao
  }
  let last  = await ultimasNoticias(7, secao)
  res.status(200).send(last)
})

app.get('/api/portais', async(req, res) => {
  let portais = await portaisDisponiveis()
  if(portais !== null) {
    res.status(200).send(portais)
  } else {
    res.status(404).send({error: "Não foi possível recuperar os portais disponíveis."})
  }
})

app.get('/api/sentimentos-tempo/:TAG', async (req, res) => {
  let tag = req.params.TAG 
  let query = req.query 
  let filtros
  try{
    filtros = processaFiltros(query)
  } catch(e) {
    res.status(404).send(e)
    return
  }
  if(filtros.campo === 'assunto.TAG') {
    tag = tag.toUpperCase()
  }
  let sentimentos = await sentimentoAtravesDoTempo(tag, filtros)
  res.status(200).send(sentimentos)
})

app.get('/api/sentimentos-gerais', async (req, res) => {
  let query = req.query, secao = ''
  if(Object.keys(query).length > 0) {
    if(Object.hasOwnProperty.bind(query)('secao'))
      secao = query.secao
  }
  let searchResult = await sentimentosGerais(7, secao)
  res.status(200).send(searchResult)
})

app.get('/api/sentimentos-portais', async (req, res) => {
  let query = req.query, secao = ''
  if(Object.keys(query).length > 0) {
    if(Object.hasOwnProperty.bind(query)('secao'))
      secao = query.secao
  }
  let searchResult = await sentimentosPortais(7, secao)
  res.status(200).send(searchResult)
})


app.get('/login', (req, res) => {
  res.sendFile(__dirname + '/frontend/login.html')
})

app.get('/register', (req, res) => {
  res.sendFile(__dirname + '/frontend/register.html')
})

// Open Route - Public Route
app.get('/', (req, res)=> {
  res.status(200).json({msg: "Bem vindo a nossa API!"})
})

// Private Route
app.get("/user/:id", checkToken, async (req, res)=>{

  const id = req.params.id
  
  // check if user exists
  const user = await User.findById(id, '-password')

  if(!user){
    return res.status(404).json({msg: 'Usuário não encontrado!'})
  }

  res.status(200).json({ user })

})

function checkToken(req, res, next){

  const authHeader = req.headers['authorization']
  const token = authHeader && authHeader.split(" ")[1]

  if(!token){
    return res.status(401).json({msg:'Acesso negado!'})
  }

  try{

    const secret = process.env.SECRET
    const decoded = jwt.verify(token, secret)
    req.userId = decoded.id // Adiciona ID do usuario ao objeto req
    next()

  } catch(error){
    res.status(400).json({msg: 'Token inválido!'})
  }
}

//Register User
app.post('/api/register', async(req, res) => {
  const {name, email, password, confirmpassword} = req.body;

  //validations
  if(!name){
    return res.status(422).json({msg: "O nome é obrigatório!"}) // Dado incorreto
  }
  if(!email){
    return res.status(422).json({msg: "O email é obrigatório!"}) // Dado incorreto
  }
  if(!password){
    return res.status(422).json({msg: "A senha é obrigatória!"}) // Dado incorreto
  }

  if(password !== confirmpassword){
    return res.status(422).json({msg: "As senhas precisam ser iguais!"}) // Dado incorreto
  }

  // check if user exist
  const userExists = await User.findOne({email: email})
  if(userExists){
    return res.status(422).json({msg: "Por favor, utilize outro e-mail"}) // Dado incorreto
  }

  // create password
  const salt = await bcrypt.genSalt(12) // adiciona dificuldade (colocar digitos a mais do que o usuario adiciona)
  const passwordHash = await bcrypt.hash(password, salt)

  // create user
  const user = new User({
    name, 
    email,
    password: passwordHash,
  })

  try {

    await user.save()

    res.status(201).json({ msg: 'Usuário criado com sucesso!'})

  } catch(error){
    console.log(error)

    res.status(500).json({
        msg: 'Tivemos um problema com o servidor, tente novamente mais tarde!',
      })

  }

})

// Login User
app.post("/api/login", async(req,res) => {

  const {email, password} = req.body

  // validations
  if(!email){
    return res.status(422).json({msg: "O e-mail é obrigatório!"}) // Dado incorreto
  }

  if(!password){
    return res.status(422).json({msg: "A senha é obrigatória!"}) // Dado incorreto
  }

  //check if user exists
  const user = await User.findOne({email: email})
  
  if(!user){
    return res.status(404).json({msg: "Usuário não encontrado!"}) // Dado incorreto
  }

  //check if password match
  const checkPassword = await bcrypt.compare(password, user.password)

  if(!checkPassword){
    return res.status(422).json({msg: "Senha incorreta!"}) // Dado incorreto
  }

  try{

    const secret = process.env.SECRET

    const token = jwt.sign({
      id: user._id
    }, 
    secret, 
   )

   res.status(200).json({msg:'Login realizado com sucesso!', token, userName: user.name})

  } catch(error) {
    console.log(error)

    res.status(500).json({
        msg: 'Tivemos um problema com o servidor, tente novamente mais tarde!',
      })
  }
})

// Route search preferences
app.get('/api/user/preferences', checkToken, async (req, res)=> {
  try {
    const userId = req.userId;
    const user = await User.findById(userId);
    
    if (!user){
      return res.status(404).json({msg: 'Usuário não encontrado.'})
    }

    res.status(200).json({ preferences: user.preferences || {} });
  } catch (error){
      console.error('Erro ao buscar preferências do usuário:', error);
      res.status(500).json({ msg: 'Erro ao carregar preferências.' });
  }
})

// Route save preferences
app.post('/api/user/preferences', checkToken, async (req, res) => {
  const { keywords } = req.body; // keywords = array strings

  if(!keywords || !Array.isArray(keywords)){
    return res.status(400).json({msg: 'As palavras-chaves devem ser fornecidas como um array.'})
  }

  try {
    const userId = req.userId; // `checkToken` deve adicionar o ID do usuário ao `req`
    const user = await User.findById(userId);

    if (!user) {
      return res.status(404).json({ msg: 'Usuário não encontrado.' });
    }

    user.preferences.keywords = keywords; // Salva as novas preferências
    user.lastNotificationCheck = new Date()
    await user.save();

    res.status(200).json({ msg: 'Preferências salvas com sucesso!' });
  } catch (error) {
      console.error('Erro ao salvar preferências do usuário:', error);
       res.status(500).json({ msg: 'Erro ao salvar preferências.' });
  }
})

// Pegar palavras-chave de um usuário
app.get('/api/user-keywords/:email', async (req, res) => {
    try {
        const userEmail = req.params.email; // Pega o email da URL
        const user = await User.findOne({ email: userEmail });

        if (!user) {
            return res.status(404).json({ message: 'Usuário não encontrado.' });
        }

        // Retorna as palavras-chave do usuário
        res.status(200).json({ keywords: user.preferences.keywords || [] });

    } catch (error) {
        console.error('Erro ao buscar palavras-chave do usuário:', error);
        res.status(500).json({ message: 'Erro interno do servidor.' });
    }
});

/*// Config nodemailer
const transporter = nodemailer.createTransport({
  service: 'outlook',
  auth: {
    user: 'politeconomy@outlook.com', // Coloque seu email aqui
    pass: 'etvkfntldvttquzl', // senha de app
  }
});*/

const transporter = nodemailer.createTransport({
     host: "smtp.mailtrap.io", // O host do Mailtrap
     port: 2525,               // A porta do Mailtrap
     auth: {
         user: "1c3eea236a975d", // Seu username do Mailtrap
         pass: "25c891e25981b4"   // Sua password do Mailtrap
     }
 });

// Send emails
async function sendNotificationEmails() {
    console.log('Iniciando verificação de novas notícias para notificações...');
    try {
        const users = await User.find({ 'preferences.keywords': { $exists: true, $not: { $size: 0 } } });

        for (const user of users) {
            const userKeywords = user.preferences.keywords;
            const lastCheckDate = user.lastNotificationCheck || new Date(0);


            if (!userKeywords || userKeywords.length === 0) {
                continue;
            }
            const regexPattern = userKeywords.map(kw => kw.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
            const combinedKeywordsRegex = new RegExp(`.*${regexPattern}.*`, 'i'); // <-- ADICIONE `.*` AQUI E AQUI!
            const queryConditions = {
                data: { $gte: lastCheckDate }, // Reative a data, que já sabemos que funciona
                $or: [ // Reative o $or e os outros campos
                    { manchete: { $regex: combinedKeywordsRegex } },
                    { lide: { $regex: combinedKeywordsRegex } },
                    { 'assunto.TAG': { $regex: combinedKeywordsRegex } }
                ]
            };
            const newRelevantNews = await Noticia.find(queryConditions)
                                                 .sort({ data: -1 })
                                                 .limit(20);

            if (newRelevantNews.length > 0) { // <-- NOVO BLOCO PARA MOSTRAR AS NOTÍCIAS ENCONTRADAS
                newRelevantNews.forEach(news => {
                });
            }


            if (newRelevantNews && newRelevantNews.length > 0) {
                let emailContent = `Olá ${user.name},\n\n`;
                emailContent += `Temos novas notícias que podem te interessar, baseadas nas suas palavras-chave (${userKeywords.join(', ')}):\n\n`;

                newRelevantNews.forEach(news => {
                    emailContent += `Manchete: ${news.manchete}\n`;
                    emailContent += `Portal: ${news.portal.nome}\n`;
                    emailContent += `Data: ${new Date(news.data).toLocaleDateString('pt-BR')}\n`;
                    emailContent += `Link: ${news._id}\n\n`;
                });

                emailContent += `Acesse nosso portal para mais informações.\n\nAtenciosamente,\nEquipe PolitEconomy`;

                const mailOptions = {
                    from: process.env.EMAIL_USER,
                    to: user.email,
                    subject: 'Novas Notícias do seu Interesse na PolitEconomy!',
                    text: emailContent
                };

                try {
                    await transporter.sendMail(mailOptions);
                    console.log(`Email enviado para ${user.email} com ${newRelevantNews.length} notícias.`);
                } catch (emailError) {
                    console.error(`Erro ao enviar email para ${user.email}:`, emailError);
                }
            } else {
                console.log(`[Notificações] Nenhuma nova notícia relevante encontrada para ${user.email}.`);
            }

            //user.lastNotificationCheck = new Date();
            await user.save(); 
        }
        console.log('Verificação de notícias finalizada.');
    } catch (error) {
        console.error('Erro na função sendNotificationEmails:', error);
    }
}

app.get('/test-send-emails', async (req, res) => {
    try {
        await sendNotificationEmails();
        res.status(200).json({ msg: 'Função de envio de e-mails acionada com sucesso!' });
    } catch (error) {
        console.error('Erro ao acionar a função de envio de e-mails:', error);
        res.status(500).json({ msg: 'Erro ao acionar a função de envio de e-mails.' });
    }
});


// INSERIR NOTICIA NO BANCO

/* app.get('/insert-test-news-from-node', async (req, res) => {
    try {
        const testNews = await Noticia.create({
            manchete: "NOTICIA TESTE DO NODE.JS:  fez um pronunciamento agora.", // Texto bem único
            lide: "Esta é uma notícia de teste inserida diretamente do aplicativo Node.js para depuração.",
            portal: { nome: "PortalTeste" },
            data: new Date(), // Data e hora atuais (garantido ser 'nova')
            autor: [{ nome: "AppTester" }],
            assunto: [{ TAG: "TESTE" }, { TAG: "PRESIDENTE" }, { TAG: "NODEJS" }],
            secao: "Depuracao"
        });
        console.log("!!! SUCESSO: NOTICIA DE TESTE INSERIDA VIA NODE.JS:", testNews._id);
        res.status(200).json({ msg: "Notícia de teste inserida com sucesso!", id: testNews._id });
    } catch (error) {
        console.error("!!! ERRO: FALHA AO INSERIR NOTÍCIA DE TESTE VIA NODE.JS:", error);
        res.status(500).json({ msg: "Erro ao inserir notícia de teste. Verifique logs do servidor." });
    }
}); */

// Agendar tarefa
cron.schedule('*/30 * * * *', () => { // Roda a cada 30 min 
//cron.schedule('* * * * *', () => { // Roda a cada 1 min 
//cron.schedule('0 0 * * *', () => { // Roda diariamente as 00h
  sendNotificationEmails()
}, {
  schedule: true,
  timezone: "America/Sao_Paulo"
})

mongoose.connect(process.env.MONGO_URI).then(()=>{ // Editei
  console.log('Conectado ao banco de dados principal (Banco_Coletor)!');
        // INICIA O SERVIDOR AQUI, DEPOIS QUE A CONEXÃO PRINCIPAL FOI ESTABELECIDA
        app.listen(PORT, () => {
            console.log(`Servidor rodando em http://localhost:${PORT}`);
            // Iniciar o cron job e o envio inicial de e-mails
            sendNotificationEmails(); 
        });
}).catch((err) => console.log(err))


process.on('SIGINT', async () => {
    console.log('Recebido sinal de encerramento (SIGINT). Desconectando do MongoDB...');
    await mongoose.disconnect(); // Desconecta apenas o Mongoose
    console.log('Conexão Mongoose desconectada. Encerrando servidor.');
    process.exit(0);
});


// Start do servidor

// src/javascript/script.js

import { mockDatabase } from './mockData.js';


export async function buscarTermo(buscaParams) {
    if(!buscaParams.hasOwnProperty('termo') || !buscaParams.hasOwnProperty('campo')) {
        console.log("Não é possível realizar busca sem especificar um termo e um campo de busca.")
        return null
    }
    let urlBase = `http://localhost:3000/api/assunto/${buscaParams.termo}?campo=${buscaParams.campo}`
    for(let i in buscaParams) {
        if(i === 'termo' || i === 'campo') continue
        urlBase = urlBase.concat(`&${i}=${buscaParams[i]}`)
    }
    let res = await fetch(urlBase)
    if(!res.ok) {
        console.log(`Houve um problema ao realizar busca sobre ${buscaParams.termo}`)
        return null
    }
    let data = await res.json()
    return data
}

export async function buscaSentimentoPorTempo(buscaParams) {
    if(!buscaParams.hasOwnProperty('termo') || !buscaParams.hasOwnProperty('campo')) {
        console.log("Não é possível realizar busca sem especificar um termo e um campo de busca.")
        return null
    }
    let urlBase = `http://localhost:3000/api/sentimentos-tempo/${buscaParams.termo}?campo=${buscaParams.campo}`
    for(let i in buscaParams) {
        if(i === 'termo' || i === 'campo') continue
        urlBase = urlBase.concat(`&${i}=${buscaParams[i]}`)
    }
    
    let res = await fetch(urlBase)
    if(!res.ok) {
        console.log(`Houve um problema ao realizar busca sobre ${buscaParams.termo}`)
        return null
    }
    let data = await res.json()
    return data
}

export function renderizarNoticias(noticias) {
    console.log("renderizarNoticias: Iniciando...");
    const noticesContainer = document.querySelector('.notices');
    if (!noticesContainer) {
        console.error("renderizarNoticias: ERRO - Elemento '.notices' não encontrado no DOM.");
        return;
    }
    console.log("renderizarNoticias: Contêiner de notícias encontrado.", noticesContainer);
    //noticesContainer.innerHTML = '';
    if (!noticias || noticias.length === 0) {
        noticesContainer.innerHTML = '<p>Nenhuma notícia encontrada.</p>';
        console.log("renderizarNoticias: Nenhuma notícia para renderizar.");
        return false;
    }
    noticias.forEach(noticia => {
        const noticeContent = document.createElement('div')
        noticeContent.className = 'notices-content'
        let sentimentClass = 'neutral';
        if (noticia.sSentimento === 1) sentimentClass = 'positive'
        else if (noticia.sSentimento === -1) sentimentClass = 'negative'
        let autores = undefined;
        if(noticia.autor && noticia.autor.length > 0) {
            autores = noticia.autor.map(x => x.nome)
        }
        noticeContent.innerHTML = `
            <div>
                <div class="sentimento-bar ${sentimentClass}"></div>
                <h3>${noticia.manchete}</h3>
                <p>Portal: ${noticia.portal.nome} - Data: ${new Date(noticia.data).toLocaleDateString('pt-BR')}</p>
                ${typeof autores !== 'undefined' && autores.length > 0 ? `<p>Autor: ${autores.join(', ')}</p>` : ''}
                <p style="font-size:0.9em; color:#333;">${noticia.lide || ''}</p>
                <a href="${noticia._id}" target="_blank" style="font-size:0.9em;">Leia mais</a>
            </div>
        `;
        noticesContainer.appendChild(noticeContent);
    });
    console.log("renderizarNoticias: Finalizado.");
    return true;
}

    // (avança e volta)
    let currentCarouselIndex = 0;
    let carouselItemsPerView = 0; 
    const carouselSpeed = 500; 

    function updateCarouselVisibility() {
        const carouselTrack = document.getElementById('latest-news-carousel');
        if (!carouselTrack) return;

        const items = carouselTrack.querySelectorAll('.carousel-item');
        if (items.length === 0) return;

        // Calcula quantos itens podem ser exibidos com base na largura do item e conteiner visivel
        const trackContainer = carouselTrack.closest('.carousel-track-container');
        if (!trackContainer) return;

        const containerWidth = trackContainer.offsetWidth;
        const itemWidth = items[0].offsetWidth; 
        const itemGap = 24;

        if (itemWidth > 0) { 
            // Calcula quantos itens cabem, considerando o gap entre eles
            carouselItemsPerView = Math.floor((containerWidth + itemGap) / (itemWidth + itemGap));
            if (carouselItemsPerView < 1) carouselItemsPerView = 1; 
        } else {
            carouselItemsPerView = 3; // Fallback
        }

        // Ajusta o deslocamento horizontal do carrossel
        // Garante que o índice esteja dentro dos limites após o redimensionamento
        if (currentCarouselIndex > items.length - carouselItemsPerView) {
            currentCarouselIndex = items.length - carouselItemsPerView;
            if (currentCarouselIndex < 0) currentCarouselIndex = 0; // Evita índice negativo se houver poucos itens
        }

        carouselTrack.style.transition = 'none'; // Desabilita transição para o ajuste de posição
        carouselTrack.style.transform = `translateX(-${currentCarouselIndex * (itemWidth + itemGap)}px)`;
        setTimeout(() => { // Reabilita transição após o ajuste
            carouselTrack.style.transition = `transform ${carouselSpeed / 1000}s ease-in-out`;
        }, 0);


        // Habilita/desabilita botões de navegação
        const prevButton = document.querySelector('.carousel-button.prev');
        const nextButton = document.querySelector('.carousel-button.next');

        if (prevButton) {
            prevButton.style.display = (currentCarouselIndex > 0) ? 'block' : 'none';
        }
        if (nextButton) {
            // Oculta o botão 'next' se já estiver exibindo os últimos itens
            nextButton.style.display = (currentCarouselIndex + carouselItemsPerView < items.length) ? 'block' : 'none';
        }
    }


    function moveCarousel(direction) {
        const carouselTrack = document.getElementById('latest-news-carousel');
        const items = carouselTrack.querySelectorAll('.carousel-item');
        if (items.length === 0) return;

        const itemWidth = items[0].offsetWidth;
        const itemGap = 24; 

        // Calcula o próximo índice
        let nextIndex = currentCarouselIndex + direction;

        // Limita o próximo índice
        if (nextIndex < 0) {
            nextIndex = 0;
        } else if (nextIndex > items.length - carouselItemsPerView) {
            nextIndex = items.length - carouselItemsPerView;
        }

        // Se o índice não mudou (atingiu o limite), não faz nada
        if (nextIndex === currentCarouselIndex) {
            return;
        }
        
        currentCarouselIndex = nextIndex;

        // Aplica o deslocamento horizontal ao contêiner
        carouselTrack.style.transform = `translateX(-${currentCarouselIndex * (itemWidth + itemGap)}px)`;

        // Atualiza a visibilidade dos botões após a transição
        setTimeout(updateCarouselVisibility, carouselSpeed);
    }


    // Adiciona a chamada para carregarUltimasNoticias no DOMContentLoaded da index.html
    document.addEventListener('DOMContentLoaded', async () => {
        const trendingTopicsContainer = document.getElementById('trending-topics-container');
        const mainContent = document.getElementById('main-content');
        const loader = document.getElementById('loader');
        try {
            if (trendingTopicsContainer) {
                await Promise.all ([
                    //carregarUltimasNoticias(), // carrossel
                    //carregarGraficoSentimentoMercado(),
                    //carregarPoliticosMencionados()
                ])

                // Adiciona event listeners para os botões do carrossel 
                const prevButton = document.querySelector('.carousel-button.prev');
                const nextButton = document.querySelector('.carousel-button.next');
                if (prevButton) {
                    prevButton.addEventListener('click', () => moveCarousel(-1));
                }
                if (nextButton) {
                    nextButton.addEventListener('click', () => moveCarousel(1));
                }
                // Atualiza a visibilidade inicial e ao redimensionar
                updateCarouselVisibility();
                window.addEventListener('resize', updateCarouselVisibility);
            }
        } catch (err) {
            console.error('Erro no carregamento inicial:', err);
        } finally {
            if (loader) loader.style.display = 'none';
            if (mainContent) mainContent.style.display = 'block';
        }
    });

    // Função auxiliar para exibir mensagens
    export function showMessage(element, message, type) {
        if (element) {
            // Limpa classes anteriores e define o texto
            element.textContent = message;
            element.className = 'message-box';
            
            // Adiciona a classe de tipo (success ou error) e a classe 'show'
            element.classList.add(type); // 'success' ou 'error'
            element.classList.add('show');

            setTimeout(() => {
                element.classList.remove('show');
                setTimeout(() => { element.textContent = ''; }, 300); 
            }, 5000); // Esconde após 5 segundos
        }
    }
    
    // --- Lógica para o Formulário de LOGIN ---
    const loginForm = document.getElementById('loginForm');
    const loginMessageElement = document.getElementById('message');

    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault(); // Impede o recarregamento da página

            // Mude para pegar apenas o email, já que userCad não será mais usado para login
            const email = document.getElementById('emailOrUserCad').value; // O ID do input do login deve ser 'email' ou 'emailOrUserCad'
            const password = document.getElementById('password').value;

            // O payload agora só precisa do email e senha
            const payload = { email, password }; // Simplificado

            try {
                const response = await fetch('http://localhost:3000/api/login', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();

                if (response.ok) {
                    showMessage(loginMessageElement, data.msg, 'success');
                    console.log('Token JWT:', data.token);
                    localStorage.setItem('jwt_token', data.token);
                    localStorage.setItem('user_name', data.userName);
                    setTimeout(() => {
                        window.location.href = 'index.html';
                    }, 1500);
                } else {
                    showMessage(loginMessageElement, `${data.msg || 'Algo deu errado!'}`, 'error');
                }
            } catch (error) {
                console.error('Erro ao conectar com a API de login:', error);
                showMessage(loginMessageElement, 'Erro ao conectar com o servidor.', 'error');
            }
        });
    }


    //Função para atualizar a navbar
    function updateNavbar() {
        const loginLinkLi = document.getElementById('navLoginLink')
        const userName = localStorage.getItem('user_name')
        const jwtToken = localStorage.getItem('jwt_token')

        if (loginLinkLi) { // Garante que o elemento exista na página
            if (jwtToken && userName) {
                loginLinkLi.innerHTML = `
                    <a href="#" class="login-link">
                        Olá, ${userName} <i class="bx bx-user"></i>
                    </a>
                    <div id="userDropdown" class="user-dropdown">
                        <button id="personalizeNotificationsButton">Personalizar Notificações</button>
                        <button id="logoutButton">Sair</button>
                    </div>
                `

                const userGreetingLink = loginLinkLi.querySelector('a')
                const userDropdown = loginLinkLi.querySelector('#userDropdown')
                const personalizeButton = loginLinkLi.querySelector('#personalizeNotificationsButton')
                const logoutButton = loginLinkLi.querySelector('#logoutButton')

                if (userGreetingLink && userDropdown && personalizeButton && logoutButton) {
                    // Clicar mostra o dropdown
                    userGreetingLink.addEventListener('click', (e) => {
                        e.preventDefault();
                        userDropdown.classList.toggle('show')
                    });

                    // Clicar fora esconde o dropdown
                    document.addEventListener('click', (e) => {
                        if (!loginLinkLi.contains(e.target) && userDropdown.classList.contains('show')) {
                            userDropdown.classList.remove('show')
                        }
                    });

                    personalizeButton.addEventListener('click', () => {
                    userDropdown.classList.remove('show'); 
                    window.location.href = 'notifications.html'
                    });

                    logoutButton.addEventListener('click', () => {
                        localStorage.removeItem('jwt_token')
                        localStorage.removeItem('user_name')
                        
                        const messageElement = document.getElementById('message'); 
                        if (messageElement) {
                            showMessage(messageElement, 'Você foi desconectado com sucesso!', 'success')
                        } else {
                            alert('Você foi desconectado!')
                        }
                        
                        // Espera um pouco antes de recarregar
                        setTimeout(() => {
                            window.location.reload()
                        }, messageElement ? 1500 : 0)
                    });
                }
            } else {
                // Se não estiver logado, exibe o link padrão "Acesse sua conta"
                loginLinkLi.innerHTML = `
                    <a href="login.html" class="login-link">Acesse sua conta <i class="bx bx-log-out"></i></a>
                `
                // Garante que qualquer dropdown anterior esteja oculto caso o estado mude para não logado
                const existingDropdown = loginLinkLi.querySelector('#userDropdown')
                if (existingDropdown) {
                    existingDropdown.remove()
                }
            }
        }
    }
    // Primeira letra do nome ser Maiusculo automaticamente
    document.addEventListener('DOMContentLoaded', () => {
    const nameInputField = document.getElementById('name'); // Pega o input do nome pelo ID

        if (nameInputField) { // Verifica se o campo de nome existe na página
            nameInputField.addEventListener('input', (e) => {
                const inputValue = e.target.value; // Pega o valor atual do input

                if (inputValue.length > 0) {
                    // Capitaliza a primeira letra e mantém o restante como está
                    e.target.value = inputValue.charAt(0).toUpperCase() + inputValue.slice(1);
                }
            });
        }
    });

    // --- Lógica para o Formulário de REGISTRO ---
    const registerForm = document.getElementById('registerForm');
    const registerMessageElement = document.getElementById('message');

    if (registerForm) {
        registerForm.addEventListener('submit', async (e) => {
            e.preventDefault(); // Impede o recarregamento da página

            const name = document.getElementById('name').value;
            const email = document.getElementById('email').value;
            const password = document.getElementById('password').value;
            const confirmpassword = document.getElementById('confirmpassword').value;
            

            try {
                const response = await fetch('http://localhost:3000/api/register', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ name, email, password, confirmpassword })
                });

                const data = await response.json();

                if (response.ok) {
                    showMessage(registerMessageElement, data.msg, 'success');
                    registerForm.reset();
                    setTimeout(() => {
                        window.location.href = 'login.html';
                    }, 1500);
                } else {
                    showMessage(registerMessageElement, `Erro: ${data.msg || 'Algo deu errado!'}`, 'error');
                }
            } catch (error) {
                console.error('Erro ao conectar com a API de registro:', error);
                showMessage(registerMessageElement, 'Erro ao conectar com o servidor.', 'error');
            }
        });
    }

document.addEventListener('DOMContentLoaded', updateNavbar);

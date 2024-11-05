<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="{{ asset('css/index.css') }}" />
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" />
    <title>Welcome to Tutur!</title>
    <!-- Include jQuery -->
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <!-- Smooth Scrolling JavaScript -->
    <script>
        $(document).ready(function() {
            $("a[href^='#']").on('click', function(event) {
                var hash = this.hash;
                if ($(hash).length) {
                    event.preventDefault();
                    $('html, body').animate({
                        scrollTop: $(hash).offset().top
                    }, 800);
                    window.location.hash = hash;
                }
            });
        });
    </script>
</head>

<body>
    <!-- Navbar -->
    <div class="navbar">
        <div class="nav-wrapper">
            <img src="{{ asset('elements/logo.png') }}" class="logo">
            <div class="nav2">
                <nav>
                    <ul>
                        <li><a href="#home">Home</a></li>
                        <li><a href="#about">About Us</a></li>
                        <li><a href="#plan">Plans</a></li>
                    </ul>
                </nav>
                <a href="{{ route('signup.page') }}" class="signup">
                    <button type="button">Sign Up <span class="material-symbols-outlined">trending_flat</span></button>
                </a>
            </div>
        </div>
    </div>
    <!-- Page 1 -->
    <div class="Home" id="home">
        <div class="home-content">

            <img src="{{ asset('elements/logo.png') }}" class="logo2">
            <img src="{{ asset('elements/home_image.png') }}" class="homeimg">
            <div class="home-cta">

                <div class="text">
                    <p>Unlock communication with sign language mastery!</p>
                    <p>Connect with deaf communities, express yourself, and bridge language barriers effortlessly.</p>
                </div>

                <span class="material-symbols-outlined">
                arrow_downward_alt
                </span>

                <a href="{{ route('signup.page') }}" class="home-button">
                    <button type="button">Join Now!</button>
                </a>
        </div>
    </div>
    </div>
    <!-- Page 2 -->
    <div class="about-us" id="about">
        <div class="about-content">
            <div class="about-col1">
                <div class="text">
                    <h1>Welcome to Tutur!</h1>
                    <p>
                        At Tutur we aim to break barriers and empower individuals worldwide to communicate fluently
                        through the universal language of signs.<br>
                        <br>
                        With interactive lessons, engaging exercises, and a supportive community, we want to make
                        learning sign language accessible, enjoyable, and effective for learners of all levels.<br>
                        <br>
                        Join us on this journey of inclusivity and connection as we bridge gaps and foster understanding
                        through the power of communication!
                    </p>
                </div>
                <a href="{{ route('signup.page') }}" class="about-button">
                    <button type="button">Get Started Learning <span class="material-symbols-outlined">trending_flat</span></button>
                </a>
            </div>
            <img src="{{ asset('elements/about_image.png') }}" alt="">
        </div>
    </div>
    <!-- Page 3 -->
    <div class="plans" id="plan">
        <div class="plans-content">
            <div class="plans-col1">
                <!-- Cards -->
                <div class="plans-card">
                    <div class="plans-card-2">
                        <div class="card-content">
                            <div class="card-row1">
                                <div class="title">Starter Pack</div>
                                <div class="price">Free</div>
                                <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque in feugiat
                                    mauris. Vivamus at ornare mi.
                                </p>
                            </div>
                            <div class="card-row2">
                                <div class="button">Details </div>
                                <span class="material-symbols-outlined">
                                    trending_flat
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
                <!-- Cards 2 -->
                <div class="plans-card-pkg2">
                    <div class="plans-card-2">
                        <div class="card-content">
                            <div class="card-row-pkg2">
                                <div class="title">Fluent Fingers</div>
                                <div class="price">IDR 60K / Mo</div>
                                <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque in feugiat
                                    mauris. Vivamus at ornare mi.
                                </p>
                            </div>
                            <div class="card-row2">
                                <div class="button"><a href="{{ route('packageDetail.page') }}">Details </a></div>
                                <span class="material-symbols-outlined">
                                    trending_flat
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
                <!-- Cards 3 -->
                <div class="plans-card-pkg3">
                    <div class="plans-card-2">
                        <div class="card-content">
                            <div class="card-row-pkg3">
                                <div class="title">Master of Motion</div>
                                <div class="price">IDR 160K / 3 Mo</div>
                                <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque in feugiat
                                    mauris. Vivamus at ornare mi.
                                </p>
                            </div>
                            <div class="card-row2">
                                <div class="button">Details </div>
                                <span class="material-symbols-outlined">
                                    trending_flat
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="plans-col2">
                <h1>Plans and Pricing</h1>
                <p>Choose Your Key to Sign Language Success: Unlock Exclusive Benefits with Our Subscription Plans!</p>
                <img src="{{ asset('elements/plans_image.png') }}" alt="">
            </div>
        </div>
    </div>
    <!-- Page 4 -->
    <div class="camera">
        <div class="camera-content">
            <div class="top">
                <h1>Tutur</h1>
                <button onclick="toggleCamera()">Preview</button>
            </div>
            <div class="camera-container">
                <img src="http://127.0.0.1:5000/video_feed" id="video" style="display: none;" alt="Live Video Stream">
            </div>
            <div class="bawah">
                <p>Here's how our Tutur App will assess your sign to ensure correct motion!</p>
                <img src="{{ asset('elements/plans_image.png') }}" alt="">
            </div>
        </div>
    </div>
    <script>
        function toggleCamera() {
            var video = document.getElementById("video");
            if (video.style.display === "none") {
                video.style.display = "block";
            } else {
                video.style.display = "none";
            }
        }
    </script>
</body>
<footer>
    <div class="foot-1">
        <div class="footer-content">
            <div class="foot-content-1">
                <div class="foot1-col1">
                    <img src="{{ asset('logo.png') }}" alt="">
                </div>
                <div class="foot1-col2">
                    <div class="col1">
                        <div class="head">Products</div>
                        <ul>
                            <li>Features</li>
                            <li>Pricing</li>
                            <li>Reviews</li>
                            <li>Updates</li>
                        </ul>
                    </div>
                    <div class="col1">
                        <div class="head">Company</div>
                        <ul>
                            <li><a href="">About</a></li>
                            <li><a href="">Contact Us</a></li>
                            <li><a href="">Careers</a></li>
                            <li><a href="">Blog</a></li>
                        </ul>
                    </div>
                    <div class="col1">
                        <div class="head">Address</div>
                        <ul>
                            <li><a href="">Level 1, 12 Sample St, <br>
                                    Sydney NSW 2000</a></li>
                        </ul>
                    </div>
                    <div class="col1">
                        <div class="head">Contact</div>
                        <ul>
                            <li><a href="">info@tutur.com</a></li>
                            <li><a href="">(62)555-0117</a></li>
                        </ul>
                    </div>
                </div>
            </div>
            <div class="divider"></div>
            <div class="foot-content-2">
                <div class="text">Copyright ©2022 Tutur. All rights reserved.</div>
                <div class="icons">
                </div>
            </div>
        </div>
    </div>
</footer>

</html>
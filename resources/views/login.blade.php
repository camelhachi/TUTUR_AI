<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="{{ asset('css/login.css') }}" />
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" />
    <title>Welcome back</title>
</head>

<body>
    <!-- Navbar -->
    <div class="navbar">
        <div class="nav-wrapper">
        <img src="{{ asset('elements/logo.png') }}" class="logo">
            <div class="nav2">
                <nav>
                    <ul>
                        <li><a href="{{ route('index.page') }}#home">Home</a></li>
                        <li><a href="{{ route('index.page') }}#about">About Us</a></li>
                        <li><a href="{{ route('index.page') }}#plan">Plans</a></li>
                    </ul>
                </nav>
                <button class="signup">Sign Up <span class="material-symbols-outlined">
                        trending_flat
                    </span></button>
            </div>
        </div>
    </div>
    <!-- Sign up box -->
    <div class="container">
        <div class="signup-form">
            <div class="signup-content">
                <div class="text">Login</div>
                <form action="{{ route('login.user') }}" method="POST">
                    @csrf
                    <label for="uname">Username</label>
                    <input type="text" id="username" name="username">
                    <label for="pwd">Password</label>
                    <input type="password" id="password" name="password">
                    <button type="submit">Login <span class="material-symbols-outlined">
                            trending_flat
                        </span></button>
                </form>
                <div class="smalltext">Don't have an account yet? <a href="{{ route('signup.page') }}">Register here.</a></div>
            </div>
        </div>
    </div>
</body>
<footer>
    <div class="foot-1">
        <div class="footer-content">
            <div class="foot-content-1">
                <div class="foot1-col1">
                    <img src="logo.png" alt="">
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

</footer>

</html>
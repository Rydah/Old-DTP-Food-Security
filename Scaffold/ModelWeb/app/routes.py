from app import application
from flask import render_template, flash, redirect, url_for,request
from app.forms import LoginForm, RegistrationForm 
from flask_login import current_user, login_user, logout_user, login_required
from app.models import User 
from werkzeug.urls import url_parse
from app import db
from flask import request 
from app.ServerLib.MultiLinearReg import * 
from app.ServerLib.PredictMLR import *
import json

@application.route('/')
@application.route('/index')
def index():
	return render_template('index.html', title='Home')

# write down your handler for the routes here


@application.route('/users')
@login_required
def users():
	users = User.query.all()	
	# mergesort(users, lambda item: item.username)
	usernames = [u.username for u in users]
	return render_template('users.html', title='Users',
							users=usernames)

@application.route('/login', methods=['GET', 'POST'])
def login():
	if current_user.is_authenticated:
		return redirect(url_for('index'))

	form = LoginForm()

	if form.validate_on_submit():#global request probably in here
		user = User.query.filter_by(username=form.username.data).first()
		if user is None or not user.check_password(form.password.data):
			flash('Invalid username or password')
			return redirect(url_for('login'))
		login_user(user, remember=form.remember_me.data)
		next_page = request.args.get('next')
		if not next_page or url_parse(next_page).netloc != '':
			next_page = url_for('index')
		return redirect(next_page)
	return render_template('login.html', title='Sign In', form=form)

@application.route('/logout')
def logout():
	logout_user()
	return redirect(url_for('index'))

@application.route('/register', methods=['GET', 'POST'])
def register():
	if current_user.is_authenticated:
		return redirect(url_for('index'))
	form = RegistrationForm()
	if form.validate_on_submit():
		user = User(username=form.username.data)
		user.set_password(form.password.data)
		db.session.add(user)
		db.session.commit()
		flash('Congratulations, you are now a registered user.')
		return redirect(url_for('login'))
	return render_template('register.html', title='Register', form=form)

@application.route('/predict',methods=['POST','GET'])
@login_required
def predict():
    if request.method=='POST':#only accept post method for form
        pass
    return render_template('predict.html',title='Predict')
        
        


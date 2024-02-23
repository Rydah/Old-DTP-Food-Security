from app import db
from app import login
from datetime import datetime 
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash


# example of how to create association table
#association_table = db.Table('association', 
#    db.Column('user_id', db.Integer, db.ForeignKey('user.id')),
#    db.Column('challenge_id', db.Integer, db.ForeignKey('challenge.id'))
#)
predset_assoctbl = db.Table('PredSetAssociation', 
    db.Column('user_id', db.Integer, db.ForeignKey('user.id')),
    db.Column('predset_id', db.Integer, db.ForeignKey('predsets.id'))
)

predres_assoctble = db.Table('PredResAssociation', 
    db.Column('predset_id', db.Integer, db.ForeignKey('predsets.id')),
    db.Column('res_id', db.Integer, db.ForeignKey('results.id'))
)#extra fat unless we serialize

@login.user_loader
def load_user(id):
	return User.query.get(int(id))

class User(UserMixin, db.Model):
	__tablename__ = 'user'
	id = db.Column(db.Integer, primary_key=True)
	username = db.Column(db.String(64), index=True, unique=True)
	password_hash = db.Column(db.String(128))
	#past_res = db.relationship("PredSet", secondary=predset_assoctbl,lazy="dynamic")
	def set_password(self, password):
		self.password_hash = generate_password_hash(password)

	def check_password(self, password):
		return check_password_hash(self.password_hash, password)

	def __repr__(self):
		return f'<User {self.username:}>'

# create your model for the database here

class Results(db.Model):
	__tablename__ = 'results'
	id = db.Column(db.Integer, primary_key=True)
	target = db.Column(db.String(64))
	setid = db.Column(db.Integer,db.ForeignKey('predsets.id'))
	res = db.Column(db.Integer)

	def __repr__(self):
		return f"<Result: {self.res}>"
     
class PredSet(db.Model):
    __tablename__= 'predsets'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    res_set = db.relationship("Results", secondary=predres_assoctble)
    feat_set = db.Column(db.String(512))

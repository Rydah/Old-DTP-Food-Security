"""empty message

Revision ID: 49d82b82bf24
Revises: 
Create Date: 2023-11-17 22:09:30.625683

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '49d82b82bf24'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('features',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('fname', sa.String(length=64), nullable=True),
    sa.Column('kname', sa.String(length=128), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('features', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_features_fname'), ['fname'], unique=True)
        batch_op.create_index(batch_op.f('ix_features_kname'), ['kname'], unique=True)

    op.create_table('user',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('username', sa.String(length=64), nullable=True),
    sa.Column('password_hash', sa.String(length=128), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_user_username'), ['username'], unique=True)

    op.create_table('predsets',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('PredSetAssociation',
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('predset_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['predset_id'], ['predsets.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], )
    )
    op.create_table('results',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('target', sa.String(length=64), nullable=True),
    sa.Column('setid', sa.Integer(), nullable=True),
    sa.Column('res', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['setid'], ['predsets.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('PredResAssociation',
    sa.Column('predset_id', sa.Integer(), nullable=True),
    sa.Column('res_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['predset_id'], ['predsets.id'], ),
    sa.ForeignKeyConstraint(['res_id'], ['results.id'], )
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('PredResAssociation')
    op.drop_table('results')
    op.drop_table('PredSetAssociation')
    op.drop_table('predsets')
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_user_username'))

    op.drop_table('user')
    with op.batch_alter_table('features', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_features_kname'))
        batch_op.drop_index(batch_op.f('ix_features_fname'))

    op.drop_table('features')
    # ### end Alembic commands ###
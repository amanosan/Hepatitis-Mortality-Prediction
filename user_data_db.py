import sqlite3

# function to create user table
def create_usertable():
    conn = sqlite3.connect("userdata.db")
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS userstable(username TEXT, password TEXT)")
    conn.close()

# function to add values into the table
def add_userdata(username, password):
    conn = sqlite3.connect("userdata.db")
    c = conn.cursor()
    c.execute("INSERT INTO userstable(username, password) VALUES (?, ?)", (username, password))
    conn.commit()
    conn.close()



# function to view the login details
def login_user(username, password):
    conn = sqlite3.connect("userdata.db")
    c = conn.cursor()
    c.execute("SELECT * FROM userstable WHERE username=? AND password=?", (username, password))
    data = c.fetchall()
    conn.close()
    return data


# function to view all the users in the table
def view_all():
    conn = sqlite3.connect("userdata.db")
    c = conn.cursor()
    c.execute("SELECT * FROM userstable")
    data = c.fetchall()
    conn.close()
    return data
